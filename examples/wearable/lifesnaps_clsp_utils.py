"""Leakage-aware preparation helpers for the LifeSnaps CLSP example.

This module is example code, not a dependency of the PyHealth model. BERT is
loaded only by ``encode_text``; synthetic tests do not download models.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from pyhealth.tasks.stress_detection_lifesnaps import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    SENSOR_CHANNELS,
)

CONTEXT_NAMES = CONTEXT_NUMERIC + CONTEXT_CATEGORICAL
STATIC_NAMES = {"age", "gender", "bmi"} | {
    name for name in CONTEXT_NAMES if name.startswith("ipip_")
}
DYNAMIC_INDICES = [
    i for i, name in enumerate(CONTEXT_NAMES) if name not in STATIC_NAMES
]


class TrainOnlyNormalizer:
    """Fit quantile transforms exclusively on the declared training partition.

    Each sensor/hour and context coordinate is mapped to [0, 1]. Context
    includes the task's deterministic category codes; this differs from the
    reference's fitted ordinal representation and is reported explicitly.
    """

    def fit(self, sensors: np.ndarray, context: np.ndarray) -> "TrainOnlyNormalizer":
        count = min(1000, len(sensors))
        self.sensor = QuantileTransformer(n_quantiles=count, random_state=0)
        self.context = QuantileTransformer(n_quantiles=count, random_state=0)
        self.sensor.fit(sensors.reshape(len(sensors), -1))
        self.context.fit(context)
        return self

    def transform(
        self, sensors: np.ndarray, context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the fitted transforms without updating their statistics."""
        ts = self.sensor.transform(sensors.reshape(len(sensors), -1))
        tab = self.context.transform(context)
        return ts.reshape(sensors.shape).astype("float32"), tab.astype("float32")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable fitted quantiles for exact inference reuse."""
        return {
            key: {
                "quantiles": model.quantiles_.tolist(),
                "references": model.references_.tolist(),
            }
            for key, model in [("sensor", self.sensor), ("context", self.context)]
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "TrainOnlyNormalizer":
        """Restore fitted transforms from the example's preprocessing JSON."""
        obj = cls()
        for key in ["sensor", "context"]:
            quantiles = np.asarray(state[key]["quantiles"])
            model = QuantileTransformer(n_quantiles=len(quantiles), random_state=0)
            model.quantiles_ = quantiles
            model.references_ = np.asarray(state[key]["references"])
            model.n_quantiles_ = len(quantiles)
            model.n_features_in_ = quantiles.shape[1]
            setattr(obj, key, model)
        return obj


def build_text(sensors: np.ndarray, context_text: str) -> str:
    """Describe non-label context and eleven raw statistics per sensor channel.

    Context goes first to retain human-readable categories when BERT's
    512-token limit truncates a long prompt. Labels and patient IDs are not
    accepted as inputs. The task supplies context_text from an allowlist.
    """
    pieces = [context_text]
    for name, series in zip(SENSOR_CHANNELS, sensors):
        std = float(np.std(series))
        standardized = (
            (series - np.mean(series)) / std if std > 0 else np.zeros_like(series)
        )
        q25, median, q75 = np.percentile(series, [25, 50, 75])
        stats = {
            "mean": np.mean(series),
            "std": std,
            "min": np.min(series),
            "max": np.max(series),
            "q25": q25,
            "median": median,
            "q75": q75,
            "rms": np.sqrt(np.mean(series**2)),
            "kurtosis": np.mean(standardized**4) - 3 if std > 0 else 0,
            "skew": np.mean(standardized**3),
            "iqr": q75 - q25,
        }
        pieces.append(name + ": " + ", ".join(f"{k} {v:.2f}" for k, v in stats.items()))
    return ". ".join(pieces).lower()


def encode_text(
    texts: list[str],
    cache_dir: Path,
    device: str,
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    local_files_only: bool = True,
    encoder_kind: str = "pretrained",
    random_seed: int = 20261125,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode with frozen BERT; cache keyed by text, model and tokenizer limit.

    A local Hugging Face model/cache is used by default. Downloads require the
    explicit example flag. Each embedding is an attention-masked mean of
    last-layer token vectors, including special tokens, with max_length=512.
    """
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if encoder_kind not in {"pretrained", "random"}:
        raise ValueError("encoder_kind must be pretrained or random")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    if encoder_kind == "pretrained":
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    else:
        config = AutoConfig.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            model = AutoModel.from_config(config)
    revision = getattr(model.config, "_commit_hash", None)
    if revision is None:
        cached_config = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached_config, str):
            revision = Path(cached_config).parent.name
    identity = {
        "texts": texts,
        "model": model_name,
        "revision": revision,
        "max_length": 512,
        "pooling": "masked-mean-with-special-tokens-v1",
    }
    if encoder_kind == "random":
        identity.update(encoder_kind=encoder_kind, random_seed=random_seed)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"bert-{key}.npz"
    if path.exists():
        stored = np.load(path, allow_pickle=False)
        metadata = json.loads(str(stored["metadata"]))
        metadata["cache_hit"] = True
        metadata["encoder_kind"] = encoder_kind
        return stored["embeddings"], metadata
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vectors = []
    truncated = 0
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        lengths = tokenizer(batch_text, truncation=False, return_length=True)["length"]
        truncated += sum(length > 512 for length in lengths)
        batch = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            vectors.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        print(f"BERT: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    embeddings = np.concatenate(vectors).astype("float32")
    if embeddings.shape != (len(texts), 768):
        raise ValueError("WearableStressModel expects 768-D BERT embeddings")
    metadata = {
        "model": model_name,
        "revision": revision,
        "max_length": 512,
        "pooling": "attention-masked mean including special tokens",
        "n_texts": len(texts),
        "truncated_texts": truncated,
        "cache_key": key,
        "cache_hit": False,
        "encoder_kind": encoder_kind,
        "random_seed": random_seed if encoder_kind == "random" else None,
    }
    np.savez_compressed(path, embeddings=embeddings, metadata=json.dumps(metadata))
    return embeddings, metadata


def statistical_features(sensors: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Return the prompt's eleven statistics per channel plus numeric context.

    No targets or participant identifiers are accepted. The downstream example
    fits scaling on its training pretraining subset and feeds these columns to
    the same trainable projection, with zero padding to its 768-D input.
    """
    sensors = np.asarray(sensors, dtype="float64")
    context = np.asarray(context, dtype="float64")
    if sensors.ndim != 3 or context.ndim != 2 or len(sensors) != len(context):
        raise ValueError("Expected aligned sensor and context batches")
    mean = sensors.mean(axis=2)
    std = sensors.std(axis=2)
    z = np.divide(
        sensors - mean[..., None],
        std[..., None],
        out=np.zeros_like(sensors),
        where=std[..., None] > 0,
    )
    q25, median, q75 = np.percentile(sensors, [25, 50, 75], axis=2)
    stats = np.stack(
        [
            mean,
            std,
            sensors.min(axis=2),
            sensors.max(axis=2),
            q25,
            median,
            q75,
            np.sqrt(np.mean(sensors**2, axis=2)),
            np.where(std > 0, np.mean(z**4, axis=2) - 3, 0),
            np.mean(z**3, axis=2),
            q75 - q25,
        ],
        axis=-1,
    ).reshape(len(sensors), -1)
    features = np.concatenate([stats, context], axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Statistical features must be finite")
    return features


def split_participants(ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Assign every participant once to fixed 60/20/20 groups without labels."""
    patients = np.unique(ids)
    if len(patients) < 4:
        raise ValueError("Need at least four participants for train/validation/test")
    patients = np.random.default_rng(seed).permutation(patients)
    a = max(1, int(0.6 * len(patients)))
    b = max(a + 1, int(0.8 * len(patients)))
    return {
        name: np.flatnonzero(np.isin(ids, group))
        for name, group in [
            ("train", patients[:a]),
            ("validation", patients[a:b]),
            ("test", patients[b:]),
        ]
    }


def calendar_context(
    context: np.ndarray,
    ids: np.ndarray,
    dates: np.ndarray,
    evaluation_rows: np.ndarray,
    delay: int,
    fallback: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Replace only dynamic context with the latest same-person date <= t-delay.

    No history uses a predeclared training-derived vector. Static attributes
    remain from the evaluation row. The evaluation rows and model weights do
    not change. Returns the substituted contexts and fallback count.
    """
    if delay < 0:
        raise ValueError("delay must be non-negative")
    result = context[evaluation_rows].copy()
    date_values = dates.astype("datetime64[D]")
    fallback_count = 0
    for j, row in enumerate(evaluation_rows):
        eligible = np.flatnonzero(
            (ids == ids[row])
            & (date_values <= date_values[row] - np.timedelta64(delay, "D"))
        )
        if len(eligible):
            source = eligible[np.argmax(date_values[eligible])]
            values = context[source]
        else:
            values = fallback
            fallback_count += 1
        result[j, DYNAMIC_INDICES] = values[DYNAMIC_INDICES]
    return result, fallback_count
