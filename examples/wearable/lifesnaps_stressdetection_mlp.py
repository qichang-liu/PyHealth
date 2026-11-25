"""LifeSnaps feature and observation-offset ablations with the existing PyHealth MLP.

Authors: Xiangwen Wang and Qichang Liu (xw120, qichang4; Illinois).
Paper: https://proceedings.mlr.press/v287/yang25a.html

Each labeled complete participant-day supplies five hourly sensor channels and
22 context features. A fixed 60/20/20 participant split is shared across all
configurations for each seed. Normalization is fitted on training samples only.
The checkpoint with the lowest validation loss is evaluated once on the test
partition; a single-class test partition has undefined ROC-AUC, reported as null.

The feature comparison retains both inputs, sensors only, or context only. The
additional context-recency experiment replaces the context vector by the same
participant's previous 1 or 7 available daily observations, with first-record
fallback. These offsets are NOT calendar days. Every configuration is retrained;
this differs from perturbing inputs to a fixed pretrained checkpoint. The offset
study investigates reliance on contemporaneous context; it does not establish a
safe questionnaire frequency or a benefit of the paper's contrastive method.

Run a software demonstration::

    python examples/wearable/lifesnaps_stressdetection_mlp.py --synthetic

Run a real-data pilot (recorded findings use the full labeled set)::

    python examples/wearable/lifesnaps_stressdetection_mlp.py \
        --root /path/to/rais_anonymized --epochs 20 --seeds 0 1 2 \
        --output /tmp/lifesnaps_mlp_results.json

Use ``--configs both --epochs 1 --seeds 0`` for a single-run timing calibration.
Actual settings and findings are in ``lifesnaps_stressdetection_findings.md``.
Synthetic metrics only test software; real MLP results are not the paper's CLSP
Table 1 results. A small fixed protocol does not establish statistical superiority.
"""

import argparse
import hashlib
import json
import logging
import platform
import tempfile
import time
from pathlib import Path
from typing import Any


def make_synthetic_root() -> str:
    """Builds a tiny synthetic rais_anonymized layout (4 participants)."""
    import numpy as np
    import pandas as pd

    root = Path(tempfile.mkdtemp())
    (root / "csv_rais_anonymized").mkdir()
    (root / "scored_surveys").mkdir()
    rng = np.random.default_rng(0)
    place = [
        "ENTERTAINMENT",
        "GYM",
        "HOME",
        "HOME_OFFICE",
        "OTHER",
        "OUTDOORS",
        "TRANSIT",
        "WORK/SCHOOL",
    ]
    rows, labels = [], []
    for p in range(1, 5):
        for d in range(1, 11):
            day = f"2021-06-{d:02d}"
            stressed = (p + d) % 2
            labels.append((f"p{p}", day, stressed))
            for h in range(24):
                rows.append(
                    {
                        "id": f"p{p}",
                        "date": day,
                        "hour": h,
                        "steps": float(rng.integers(0, 500)) * (1 + stressed),
                        "bpm": float(rng.integers(55, 100)) + 10 * stressed,
                        "calories": float(rng.integers(50, 300)),
                        "distance": float(rng.random()),
                        "temperature": 33.0 + rng.random(),
                        "minutes_in_default_zone_1": 5 + 2 * stressed + d,
                        "minutes_below_default_zone_1": 40,
                        "minutes_in_default_zone_2": 2,
                        "minutes_in_default_zone_3": 0,
                        "age": "<30",
                        "gender": "MALE",
                        "bmi": "<25",
                        "step_goal_label": "below goal",
                        **{c: 0 for c in place},
                        "activityType": np.nan,
                    }
                )
    pd.DataFrame(rows).to_csv(
        root / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "user_id": [f"p{p}" for p in range(1, 5)],
            **{
                c: ["Average"] * 4
                for c in [
                    "ipip_extraversion_category",
                    "ipip_agreeableness_category",
                    "ipip_conscientiousness_category",
                    "ipip_stability_category",
                    "ipip_intellect_category",
                ]
            },
        }
    ).to_csv(root / "scored_surveys" / "personality.csv", index=False)
    pd.DataFrame(
        {
            "user_id": [r[0] for r in labels],
            "submitdate": [f"{r[1]} 20:00:00" for r in labels],
            "stai_stress": [60 if r[2] else 30 for r in labels],
            "stai_stress_category": [
                "Above average" if r[2] else "Below average" for r in labels
            ],
        }
    ).to_csv(root / "scored_surveys" / "stai.csv", index=False)
    return str(root)


def make_flat_task(**task_kwargs: Any) -> Any:
    """StressDetectionLifeSnaps variant with sensors flattened to 120-d.

    ``pyhealth.models.MLP`` flattens 3D float inputs before its embedding
    layer (it targets code sequences), which conflicts with the (5, 24)
    sensor tensor. Flattening in the task keeps the example on the stock
    MLP; models that consume (channels, time) tensors can use the canonical
    task directly.
    """
    from pyhealth.tasks import StressDetectionLifeSnaps

    class FlatStressDetectionLifeSnaps(StressDetectionLifeSnaps):
        task_name = "StressDetectionLifeSnapsFlat"

        def __call__(self, patient: Any) -> list[dict[str, Any]]:
            samples = super().__call__(patient)
            for sample in samples:
                sample["sensors"] = [
                    v for channel in sample["sensors"] for v in channel
                ]
            return samples

    return FlatStressDetectionLifeSnaps(**task_kwargs)


def _partition_summary(samples: Any) -> dict[str, Any]:
    """Describe a split without publishing participant identifiers."""
    import torch

    rows = [samples[i] for i in range(len(samples))]
    keys = sorted((str(row["patient_id"]), str(row["day"])) for row in rows)
    positives = sum(int(torch.as_tensor(row["stress"]).item()) for row in rows)
    return {
        "samples": len(rows),
        "patients": len({key[0] for key in keys}),
        "positive": positives,
        "negative": len(rows) - positives,
        "sample_key_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
    }


def _normalized_loaders(parts: tuple, batch_size: int = 64) -> list:
    """Fit per-feature moments on training samples, then transform all splits."""
    import torch
    from torch.utils.data import Dataset

    from pyhealth.datasets import get_dataloader

    if any(len(part) == 0 for part in parts):
        raise ValueError("Train, validation and test partitions must be nonempty")
    train = parts[0]
    stats = {}
    for key in ("sensors", "context"):
        values = torch.stack(
            [torch.as_tensor(train[i][key]).float() for i in range(len(train))]
        )
        if not torch.isfinite(values).all():
            raise ValueError(f"Nonfinite training values in {key}")
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        stats[key] = (mean, scale)

    class NormalizedSamples(Dataset):
        def __init__(self, source: Any) -> None:
            self.source = source

        def __len__(self) -> int:
            return len(self.source)

        def set_shuffle(self, shuffle: bool) -> None:
            self.source.set_shuffle(shuffle)

        def __getitem__(self, index: int) -> dict:
            item = dict(self.source[index])
            for key, (mean, scale) in stats.items():
                value = (torch.as_tensor(item[key]).float() - mean) / scale
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite normalized values in {key}")
                item[key] = value
            return item

    return [
        get_dataloader(
            NormalizedSamples(part), batch_size=batch_size, shuffle=index == 0
        )
        for index, part in enumerate(parts)
    ]


def _binary_metrics(y_true: Any, y_prob: Any, loss: float) -> dict[str, Any]:
    """Report probability ROC-AUC, explicitly undefined for a single class."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true, y_prob = np.asarray(y_true).ravel(), np.asarray(y_prob).ravel()
    if y_true.size == 0 or y_true.shape != y_prob.shape:
        raise ValueError("Expected matching, nonempty labels and probabilities")
    if not np.isfinite(y_prob).all() or not np.isfinite(loss):
        raise ValueError("Nonfinite test loss or probabilities")
    auc = float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size == 2 else None
    return {
        "test_roc_auc": auc,
        "test_roc_auc_reason": None
        if auc is not None
        else "single-class test partition",
        "test_loss": float(loss),
    }


def run_config(
    dataset: Any, label: str, epochs: int, seed: int, **task_kwargs: Any
) -> dict[str, Any]:
    """Train one prespecified configuration and report held-out probabilities.

    Args:
        dataset: Prepared LifeSnapsDataset; reuse it across configurations.
        label: Configuration identifier used in the output record.
        epochs: Positive maximum training epochs.
        seed: Initialization, batch shuffle and participant-split seed.
        **task_kwargs: Parameters of StressDetectionLifeSnaps.

    Returns:
        Configuration, split counts, timings, test loss and probability ROC-AUC.
    """
    from pyhealth.datasets import PatientSplit
    from pyhealth.models import MLP
    from pyhealth.trainer import Trainer
    from pyhealth.utils import set_seed

    start = time.perf_counter()
    set_seed(seed)
    parts = dataset.set_task(
        make_flat_task(**task_kwargs),
        split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=seed),
    )
    summaries = dict(
        zip(("train", "validation", "test"), map(_partition_summary, parts))
    )
    patient_sets = [
        {str(part[i]["patient_id"]) for i in range(len(part))} for part in parts
    ]
    if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i)):
        raise AssertionError("A participant crossed the split boundary")
    if summaries["train"]["positive"] == 0 or summaries["train"]["negative"] == 0:
        raise ValueError("Training partition needs both stress classes")
    train_loader, val_loader, test_loader = _normalized_loaders(parts)
    model = MLP(dataset=parts[0], embedding_dim=32, hidden_dim=32, n_layers=2)
    prepare_seconds = time.perf_counter() - start
    train_start = time.perf_counter()
    trainer_logger = logging.getLogger("pyhealth.trainer")
    old_handlers = set(trainer_logger.handlers)
    with tempfile.TemporaryDirectory(prefix="lifesnaps-mlp-model-") as checkpoint_dir:
        try:
            trainer = Trainer(
                model=model,
                metrics=[],
                device="cpu",
                output_path=checkpoint_dir,
                exp_name="run",
            )
            trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=epochs,
                monitor="loss",
                monitor_criterion="min",
                optimizer_params={"lr": 1e-3},
                load_best_model_at_last=True,
            )
            train_seconds = time.perf_counter() - train_start
            y_true, y_prob, loss = trainer.inference(test_loader)
            metrics = _binary_metrics(y_true, y_prob, loss)
        finally:
            for handler in set(trainer_logger.handlers) - old_handlers:
                trainer_logger.removeHandler(handler)
                handler.close()
    result = {
        "configuration": label,
        "seed": seed,
        "epochs": epochs,
        "task_kwargs": task_kwargs,
        "partitions": summaries,
        **metrics,
        "prepare_seconds": prepare_seconds,
        "train_seconds": train_seconds,
        "total_seconds": time.perf_counter() - start,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    """Run fixed ablations and optionally export aggregate, identifier-free JSON."""
    import torch

    from pyhealth.datasets import LifeSnapsDataset

    configs = {
        "both": {},
        "sensors": {"feature_set": "sensors"},
        "context": {"feature_set": "context"},
        "offset1": {"context_staleness_days": 1},
        "offset7": {"context_staleness_days": 7},
    }
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="path to extracted rais_anonymized")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--configs", choices=list(configs), nargs="+", default=list(configs)
    )
    parser.add_argument(
        "--output", type=Path, help="JSON path; no patient IDs included"
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(1)
    synthetic_root = make_synthetic_root() if args.synthetic else None
    root = synthetic_root or args.root
    try:
        with tempfile.TemporaryDirectory(prefix="lifesnaps-mlp-cache-") as cache:
            dataset = LifeSnapsDataset(root=root, cache_dir=cache)
            results = [
                run_config(
                    dataset,
                    name,
                    min(args.epochs, 3) if args.synthetic else args.epochs,
                    seed,
                    **configs[name],
                )
                for seed in args.seeds
                for name in args.configs
            ]
        # Same participant-day partitions are required for every feature/offset
        # configuration. A changed task must not silently change the comparison.
        for seed in args.seeds:
            rows = [row for row in results if row["seed"] == seed]
            for part in ("train", "validation", "test"):
                fingerprints = {
                    row["partitions"][part]["sample_key_sha256"] for row in rows
                }
                if len(fingerprints) != 1:
                    raise AssertionError(
                        "Configurations used different evaluation rows"
                    )
        payload = {
            "synthetic": args.synthetic,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
            "split": "PatientSplit(0.6, 0.2, 0.2); per-seed fixed across configurations",
            "normalization": "per-coordinate training mean/std; constant scale=1",
            "selection": "minimum validation BCE loss; no test selection",
            "model": "PyHealth MLP; embedding_dim=32, hidden_dim=32, n_layers=2",
            "optimizer": "Adam lr=0.001, batch_size=64",
            "source_sha256": {
                str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(__file__),
                    Path(__file__).parents[2] / "pyhealth/datasets/lifesnaps.py",
                    Path(__file__).parents[2]
                    / "pyhealth/tasks/stress_detection_lifesnaps.py",
                ]
            },
            "runs": results,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print("=== Held-out participant probability ROC-AUC ===")
        for row in results:
            value = row["test_roc_auc"]
            print(
                f"seed={row['seed']} {row['configuration']:8s}: "
                + (f"{value:.4f}" if value is not None else "undefined (single class)")
            )
    finally:
        if synthetic_root:
            import shutil

            shutil.rmtree(synthetic_root)


if __name__ == "__main__":
    main()
