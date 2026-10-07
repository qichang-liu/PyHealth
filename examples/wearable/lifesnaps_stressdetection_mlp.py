"""Ablation study: LifeSnaps + StressDetectionLifeSnaps + PyHealth MLP.

Author: Qichang Liu (NetID: TODO, liuqc14@gmail.com)
Paper: Yang, Yu & Sano, "Contrastive Pretraining for Stress Detection with
Multimodal Wearable Sensor Data and Surveys", CHIL 2025, PMLR v287.
https://proceedings.mlr.press/v287/yang25a.html

Experimental setup
------------------
One sample = one labeled participant-day: hourly sensors (5, 24) + survey/
context vector -> binary stress. We train the existing ``pyhealth.models.MLP``
with a patient-level split (no participant crosses splits) and compare task
configurations:

1. feature-set ablation: both modalities vs sensors-only vs context-only;
2. survey-staleness ablation (NOVEL, absent from the paper): the context
   vector comes from the participant's day t-k (k in {0, 1, 7}).

NOTE: results produced with this script use a PyHealth MLP, not the paper's
contrastive architecture — they must not be read as a reproduction of the
paper's Table 1 (that reproduction lives outside PyHealth).

Run on real data:    python lifesnaps_stressdetection_mlp.py --root /path/to/rais_anonymized
Run a synthetic demo: python lifesnaps_stressdetection_mlp.py --synthetic

Findings template (fill from a real run): with both modalities the MLP
reaches a test ROC-AUC well above chance; sensors-only typically drops a few
points; context staleness degrades AUC gradually with k.
"""

import argparse
import tempfile
from pathlib import Path


def make_synthetic_root() -> str:
    """Builds a tiny synthetic rais_anonymized layout (4 participants)."""
    import numpy as np
    import pandas as pd

    root = Path(tempfile.mkdtemp())
    (root / "csv_rais_anonymized").mkdir()
    (root / "scored_surveys").mkdir()
    rng = np.random.default_rng(0)
    place = ["ENTERTAINMENT", "GYM", "HOME", "HOME_OFFICE",
             "OTHER", "OUTDOORS", "TRANSIT", "WORK/SCHOOL"]
    rows, labels = [], []
    for p in range(1, 5):
        for d in range(1, 11):
            day = f"2021-06-{d:02d}"
            stressed = int(rng.random() < 0.4)
            labels.append((f"p{p}", day, stressed))
            for h in range(24):
                rows.append({
                    "id": f"p{p}", "date": day, "hour": h,
                    "steps": float(rng.integers(0, 500)) * (1 + stressed),
                    "bpm": float(rng.integers(55, 100)) + 10 * stressed,
                    "calories": float(rng.integers(50, 300)),
                    "distance": float(rng.random()),
                    "temperature": 33.0 + rng.random(),
                    "minutes_in_default_zone_1": 5,
                    "minutes_below_default_zone_1": 40,
                    "minutes_in_default_zone_2": 2,
                    "minutes_in_default_zone_3": 0,
                    "age": "<30", "gender": "MALE", "bmi": "<25",
                    "step_goal_label": "below goal",
                    **{c: 0 for c in place},
                    "activityType": np.nan,
                })
    pd.DataFrame(rows).to_csv(
        root / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv",
        index=False,
    )
    pd.DataFrame({
        "user_id": [f"p{p}" for p in range(1, 5)],
        **{c: ["Average"] * 4 for c in [
            "ipip_extraversion_category", "ipip_agreeableness_category",
            "ipip_conscientiousness_category", "ipip_stability_category",
            "ipip_intellect_category"]},
    }).to_csv(root / "scored_surveys" / "personality.csv", index=False)
    pd.DataFrame({
        "user_id": [r[0] for r in labels],
        "submitdate": [f"{r[1]} 20:00:00" for r in labels],
        "stai_stress": [60 if r[2] else 30 for r in labels],
        "stai_stress_category": [
            "Above average" if r[2] else "Below average" for r in labels
        ],
    }).to_csv(root / "scored_surveys" / "stai.csv", index=False)
    return str(root)


def make_flat_task(**task_kwargs):
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

        def __call__(self, patient):
            samples = super().__call__(patient)
            for sample in samples:
                sample["sensors"] = [
                    v for channel in sample["sensors"] for v in channel
                ]
            return samples

    return FlatStressDetectionLifeSnaps(**task_kwargs)


def run_config(root: str, label: str, epochs: int, **task_kwargs) -> dict:
    from pyhealth.datasets import (
        LifeSnapsDataset,
        get_dataloader,
        split_by_patient,
    )
    from pyhealth.models import MLP
    from pyhealth.trainer import Trainer

    dataset = LifeSnapsDataset(root=root)
    samples = dataset.set_task(make_flat_task(**task_kwargs))
    train_ds, val_ds, test_ds = split_by_patient(samples, [0.6, 0.2, 0.2])
    train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)
    model = MLP(dataset=samples)
    trainer = Trainer(model=model, metrics=["roc_auc"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="roc_auc",
    )
    result = trainer.evaluate(test_loader)
    print(f"[{label}] test: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="path to unzipped rais_anonymized")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    root = make_synthetic_root() if args.synthetic else args.root
    if root is None:
        raise SystemExit("pass --root <rais_anonymized dir> or --synthetic")
    epochs = 3 if args.synthetic else args.epochs

    configs = [
        ("both-modalities (k=0)", {}),
        ("sensors-only", {"feature_set": "sensors"}),
        ("context-only", {"feature_set": "context"}),
        ("stale-context k=1", {"context_staleness_days": 1}),
        ("stale-context k=7", {"context_staleness_days": 7}),
    ]
    results = {}
    for label, kwargs in configs:
        results[label] = run_config(root, label, epochs, **kwargs)

    print("\n=== Ablation summary (test ROC-AUC) ===")
    for label, res in results.items():
        print(f"{label:24s} {res.get('roc_auc', float('nan')):.4f}")


if __name__ == "__main__":
    main()
