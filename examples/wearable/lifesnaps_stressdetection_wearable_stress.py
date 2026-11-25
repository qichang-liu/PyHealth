"""LifeSnaps -> Task -> CLSP pretraining -> probe -> fine-tune -> evaluation.

Authors: Xiangwen Wang (xw120@illinois.edu), Qichang Liu (qichang4@illinois.edu).
Paper: https://proceedings.mlr.press/v287/yang25a.html

The example compares supervised training with text-guided pretraining and
performs a new fixed-weight calendar-day context delay intervention. It uses
participant-level 60/20/20 splits, not the paper's random-day Table 1 protocol.
Only training participants enter pretraining and preprocessing fitting.
BERT is frozen and processes allowed context text plus sensor statistics.

Offline smoke test (random text vectors, NOT a scientific result)::

    python examples/wearable/lifesnaps_stressdetection_wearable_stress.py \
        --synthetic --output /tmp/lifesnaps-smoke

Real-data calibration before a larger experiment (cached BERT required)::

    python examples/wearable/lifesnaps_stressdetection_wearable_stress.py \
        --root /path/to/rais_anonymized --output /tmp/lifesnaps-calibration \
        --pretrain-epochs 1 --probe-epochs 1 --finetune-epochs 1 --supervised-epochs 1

The declared default training budget is smaller than the paper's budget;
passing 300/150/150 and supervised 300 reproduces its epoch counts, not its
split/feature implementation. The model uses one binary logit and the task's
category encoding; numerical equivalence to the reference is not claimed.
The default reference objective differentiates through its soft targets.
Use --contrastive-target stop_gradient for the earlier detached adaptation,
or one_hot to isolate paired one-hot targets with all other settings fixed.
Outputs include metrics.json, preprocessing.json, split.json and model
checkpoints. Findings belong in the accompanying result Markdown after an
actual run. No clinical conclusion should be drawn from synthetic scores.
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lifesnaps_clsp_utils import (
    TrainOnlyNormalizer,
    build_text,
    calendar_context,
    encode_text,
    split_participants,
    statistical_features,
)
from lifesnaps_stressdetection_mlp import make_synthetic_root
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import LifeSnapsDataset, create_sample_dataset
from pyhealth.models import WearableStressModel
from pyhealth.tasks import StressDetectionLifeSnaps
from pyhealth.utils import set_seed


def metrics(
    model: WearableStressModel,
    ts: np.ndarray,
    tab: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate frozen weights on the supplied rows without fitting anything."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(ts), batch_size):
            out = model(
                sensors=torch.as_tensor(ts[start : start + batch_size]),
                context=torch.as_tensor(tab[start : start + batch_size]),
            )
            predictions.extend(out["y_prob"].flatten().cpu().tolist())
    prob = np.asarray(predictions)
    both_classes = len(np.unique(labels)) == 2
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, prob)) if both_classes else None,
        "hard_label_auc": float(roc_auc_score(labels, prob >= 0.5))
        if both_classes
        else None,
        "probabilities": prob.tolist(),
    }


def train_stage(
    model: WearableStressModel,
    stage: str,
    ts: np.ndarray,
    tab: np.ndarray,
    target: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Optimize a stage; supervised stages restore their best validation AUC."""
    model.set_stage(stage)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    rng = np.random.default_rng(seed)
    history = []
    best_score = -float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(ts))
        losses = []
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            if stage == "pretrain" and len(rows) < 2:
                continue
            batch = {
                "sensors": torch.as_tensor(ts[rows]),
                "context": torch.as_tensor(tab[rows]),
                "text_emb" if stage == "pretrain" else "stress": torch.as_tensor(
                    target[rows]
                ),
            }
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss during {stage}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        if validation is not None:
            score = metrics(model, *validation)["roc_auc"]
            row["validation_auc"] = score
            if score is not None and score > best_score:
                best_score = score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        history.append(row)
        print(f"{stage} {epoch + 1}/{epochs}: {row}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def run(args: argparse.Namespace, root: str) -> dict[str, Any]:
    """Execute a complete paired-text pipeline and save auditable artifacts."""
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = LifeSnapsDataset(root=root, cache_dir=args.output / "dataset-cache")
    labeled = dataset.set_task(StressDetectionLifeSnaps())
    complete = dataset.set_task(StressDetectionLifeSnaps(pretraining=True))
    records = [dict(complete[i]) for i in range(len(complete))]
    if not records:
        raise ValueError("No complete days available")
    label_map = {
        (str(s["patient_id"]), str(s["day"])): int(torch.as_tensor(s["stress"]).item())
        for s in (labeled[i] for i in range(len(labeled)))
    }
    ids = np.asarray([str(s["patient_id"]) for s in records])
    dates = np.asarray([str(s["day"]) for s in records])
    raw_ts = np.stack([np.asarray(s["sensors"], dtype="float32") for s in records])
    raw_tab = np.stack([np.asarray(s["context"], dtype="float32") for s in records])
    y = np.asarray([label_map.get((pid, day), -1) for pid, day in zip(ids, dates)])
    split = split_participants(ids, args.seed)
    train_rows = split["train"]
    pretrain_rows = train_rows
    if args.max_pretrain_days and len(train_rows) > args.max_pretrain_days:
        pretrain_rows = np.sort(
            np.random.default_rng(args.seed).choice(
                train_rows, size=args.max_pretrain_days, replace=False
            )
        )
    supervised_rows = {key: rows[y[rows] >= 0] for key, rows in split.items()}
    for name, rows in supervised_rows.items():
        if not len(rows):
            raise ValueError(f"Fixed participant split has no labeled {name} rows")
        if not args.synthetic and len(np.unique(y[rows])) != 2:
            raise ValueError(
                f"Fixed {name} split has one class; report and redesign split, do not select on test AUC"
            )
    split_record = {
        key: {
            "participants": sorted(set(ids[rows].tolist())),
            "complete_days": len(rows),
            "labeled_days": len(supervised_rows[key]),
            "positives": int(y[supervised_rows[key]].sum()),
        }
        for key, rows in split.items()
    }
    (args.output / "split.json").write_text(json.dumps(split_record, indent=2))
    normalizer = TrainOnlyNormalizer().fit(raw_ts[train_rows], raw_tab[train_rows])
    ts, tab = normalizer.transform(raw_ts, raw_tab)
    normalizer_state = normalizer.to_dict()
    (args.output / "preprocessing.json").write_text(json.dumps(normalizer_state))
    # Roundtrip fitted preprocessing before any reported evaluation.
    restored_ts, restored_tab = TrainOnlyNormalizer.from_dict(
        normalizer_state
    ).transform(raw_ts, raw_tab)
    np.testing.assert_allclose(restored_ts, ts)
    np.testing.assert_allclose(restored_tab, tab)
    preprocessing_seconds = time.perf_counter() - started
    text_started = time.perf_counter()
    if args.synthetic:
        embeddings = (
            np.random.default_rng(args.seed)
            .normal(size=(len(pretrain_rows), 768))
            .astype("float32")
        )
        text_meta = {"kind": "synthetic random vectors; software smoke only"}
    elif args.text_encoder == "statistics":
        features = statistical_features(raw_ts[pretrain_rows], raw_tab[pretrain_rows])
        mean, scale = features.mean(0), features.std(0)
        scale[scale == 0] = 1
        scaled = ((features - mean) / scale).astype("float32")
        embeddings = np.pad(scaled, ((0, 0), (0, 768 - scaled.shape[1])))
        text_meta = {
            "encoder_kind": "statistics",
            "n_texts": len(features),
            "feature_dim": features.shape[1],
            "projection": "trainable existing Linear(768,128); unused inputs zero",
            "scaling": "mean/std fitted only on selected training pretraining days",
        }
        (args.output / "statistical_preprocessing.json").write_text(
            json.dumps({"mean": mean.tolist(), "scale": scale.tolist()})
        )
    else:
        texts = [
            build_text(raw_ts[row], str(records[row]["context_text"]))
            for row in pretrain_rows
        ]
        embeddings, text_meta = encode_text(
            texts,
            args.text_cache or args.output / "text-cache",
            device,
            args.text_model,
            args.text_batch_size,
            local_files_only=not args.allow_download,
            encoder_kind=args.text_encoder,
            random_seed=args.random_encoder_seed,
        )
    text_seconds = time.perf_counter() - text_started
    tr = supervised_rows["train"]
    va = supervised_rows["validation"]
    te = supervised_rows["test"]
    schema_samples = [
        {
            "patient_id": ids[row],
            "sensors": ts[row].tolist(),
            "context": tab[row].tolist(),
            "stress": int(y[row]),
        }
        for row in tr
    ]
    model_dataset = create_sample_dataset(
        schema_samples, {"sensors": "tensor", "context": "tensor"}, {"stress": "binary"}
    )
    result: dict[str, Any] = {
        "synthetic": args.synthetic,
        "seed": args.seed,
        "device": device,
        "split": split_record,
        "text": text_meta,
        "protocol": "participant 60/20/20; preprocessing and pretraining fit only train participants",
        "normalization": "coordinate quantiles of sensor/hour and context, including task category codes",
        "timing_seconds": {
            "preprocessing": preprocessing_seconds,
            "text": text_seconds,
        },
        "models": {},
        "pretraining_days": len(pretrain_rows),
        "evaluation": {
            "patient_ids": ids[te].tolist(),
            "dates": dates[te].tolist(),
            "labels": y[te].tolist(),
        },
    }
    fallback = np.median(raw_tab[train_rows], axis=0)
    for name in ["supervised", "clsp"]:
        set_seed(args.seed)
        model = WearableStressModel(
            model_dataset,
            dropout=args.dropout,
            contrastive_target=args.contrastive_target,
        ).to(device)
        training_started = time.perf_counter()
        history = {}
        if name == "supervised":
            history["supervised"] = train_stage(
                model,
                "supervised",
                ts[tr],
                tab[tr],
                y[tr],
                args.supervised_epochs,
                args.lr,
                args.batch_size,
                args.seed,
                validation=(ts[va], tab[va], y[va]),
            )
        else:
            history["pretrain"] = train_stage(
                model,
                "pretrain",
                ts[pretrain_rows],
                tab[pretrain_rows],
                embeddings,
                args.pretrain_epochs,
                args.lr,
                args.batch_size,
                args.seed,
            )
            history["probe"] = train_stage(
                model,
                "probe",
                ts[tr],
                tab[tr],
                y[tr],
                args.probe_epochs,
                args.lr,
                args.batch_size,
                args.seed + 1,
                validation=(ts[va], tab[va], y[va]),
            )
            history["finetune"] = train_stage(
                model,
                "finetune",
                ts[tr],
                tab[tr],
                y[tr],
                args.finetune_epochs,
                args.lr,
                args.batch_size,
                args.seed + 2,
                validation=(ts[va], tab[va], y[va]),
            )
        training_seconds = time.perf_counter() - training_started
        checkpoint = args.output / f"{name}.pt"
        model.save_checkpoint(checkpoint)
        reference = metrics(model, ts[te], tab[te], y[te])
        restored = WearableStressModel.load_checkpoint(checkpoint, model_dataset).to(
            device
        )
        restored_metrics = metrics(restored, ts[te], tab[te], y[te])
        np.testing.assert_allclose(
            reference["probabilities"], restored_metrics["probabilities"], atol=1e-6
        )
        ablations = []
        for delay in args.delays:
            contexts, fallback_count = calendar_context(
                raw_tab, ids, dates, te, delay, fallback
            )
            _, transformed = normalizer.transform(raw_ts[te], contexts)
            score = metrics(restored, ts[te], transformed, y[te])
            score.update({"calendar_days": delay, "fallback_count": fallback_count})
            ablations.append(score)
        result["models"][name] = {
            "training_seconds": training_seconds,
            "history": history,
            "test": reference,
            "calendar_context_ablation": ablations,
            "checkpoint_roundtrip": "predictions agree within absolute tolerance 1e-6",
        }
        print(f"{name}: test probability AUC={reference['roc_auc']}", flush=True)
    result["timing_seconds"]["total"] = time.perf_counter() - started
    result["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    return result


def main() -> None:
    """Parse explicit data, model-cache, training-budget, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--text-model", default="bert-base-uncased")
    parser.add_argument(
        "--text-encoder",
        choices=["pretrained", "random", "statistics"],
        default="pretrained",
    )
    parser.add_argument("--random-encoder-seed", type=int, default=20261125)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument(
        "--max-pretrain-days",
        type=int,
        default=0,
        help="0 uses all training days; otherwise a seeded training-only subset",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--supervised-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--contrastive-target",
        choices=["reference", "stop_gradient", "one_hot"],
        default="reference",
        help="Reference differentiable soft targets, legacy detached targets, or paired indices",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 1, 3, 7])
    args = parser.parse_args()
    if not args.synthetic and not args.root:
        parser.error("pass --root or --synthetic")
    if args.max_pretrain_days < 0 or args.max_pretrain_days == 1:
        parser.error("max-pretrain-days must be 0 or >= 2")
    if (
        any(
            getattr(args, key) < 1
            for key in [
                "pretrain_epochs",
                "probe_epochs",
                "finetune_epochs",
                "supervised_epochs",
                "text_batch_size",
            ]
        )
        or args.batch_size < 2
    ):
        parser.error("epochs/text-batch-size must be positive; batch-size >= 2")
    if args.synthetic:
        for key in [
            "pretrain_epochs",
            "probe_epochs",
            "finetune_epochs",
            "supervised_epochs",
        ]:
            setattr(args, key, 1)
    root = make_synthetic_root() if args.synthetic else args.root
    try:
        result = run(args, root)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "timing_seconds": result["timing_seconds"],
                },
                indent=2,
            )
        )
    finally:
        if args.synthetic:
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
