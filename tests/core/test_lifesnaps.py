"""Tests for LifeSnapsDataset and StressDetectionLifeSnaps (synthetic data).

Author: Qichang Liu (NetID: TODO, liuqc14@gmail.com)
Paper: https://proceedings.mlr.press/v287/yang25a.html
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import LifeSnapsDataset
from pyhealth.tasks import StressDetectionLifeSnaps
from pyhealth.tasks.stress_detection_lifesnaps import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    EXCLUDED_FEATURES,
)

PLACE = ["ENTERTAINMENT", "GYM", "HOME", "HOME_OFFICE",
         "OTHER", "OUTDOORS", "TRANSIT", "WORK/SCHOOL"]


class TestLifeSnaps(unittest.TestCase):
    """Synthetic 4-participant fixture exercising dataset + task."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        root = Path(cls.temp_dir)
        (root / "csv_rais_anonymized").mkdir()
        (root / "scored_surveys").mkdir()
        rng = np.random.default_rng(0)

        rows = []
        # p1: 2 complete days; p2: 1 complete day; p3: INCOMPLETE day (23h);
        # p4: complete day with missing sensor values and shuffled-hour dup.
        plan = {
            "p1": [("2021-06-01", 24), ("2021-06-02", 24)],
            "p2": [("2021-06-01", 24)],
            "p3": [("2021-06-01", 23)],
            "p4": [("2021-06-03", 24)],
        }
        for pid, days in plan.items():
            for day, n_hours in days:
                for h in range(n_hours):
                    rows.append({
                        "id": pid, "date": day, "hour": h,
                        "steps": np.nan if (pid == "p4" and h < 3)
                        else float(rng.integers(0, 500)),
                        "bpm": float(rng.integers(55, 110)),
                        "calories": float(rng.integers(50, 300)),
                        "distance": float(rng.random()),
                        "temperature": 33.0 + rng.random(),
                        "minutes_in_default_zone_1": 5,
                        "minutes_below_default_zone_1": 40,
                        "minutes_in_default_zone_2": 2,
                        "minutes_in_default_zone_3": 0,
                        "age": "<30", "gender": "MALE", "bmi": "<25",
                        "step_goal_label": "below goal",
                        **{c: 0 for c in PLACE},
                        "activityType": np.nan,
                    })
        # duplicate-hour row for p4 (dataset must de-duplicate, keep 24)
        rows.append(dict(rows[-1], hour=23))
        pd.DataFrame(rows).to_csv(
            root / "csv_rais_anonymized"
            / "hourly_fitbit_sema_df_unprocessed.csv",
            index=False,
        )

        pd.DataFrame({
            "user_id": ["p1", "p2", "p3", "p4"],
            "ipip_extraversion_category": ["High", "Low", "Average", "High"],
            "ipip_agreeableness_category": ["Average"] * 4,
            "ipip_conscientiousness_category": ["Low"] * 4,
            "ipip_stability_category": ["High"] * 4,
            "ipip_intellect_category": ["Average"] * 4,
        }).to_csv(root / "scored_surveys" / "personality.csv", index=False)

        # labels: p1 day1 stressed, p1 day2 relaxed, p2 day1 relaxed,
        # p3 labeled but its day is incomplete -> must NOT become a sample;
        # p4 unlabeled.
        pd.DataFrame({
            "user_id": ["p1", "p1", "p2", "p3"],
            "submitdate": ["2021-06-01 20:00:00", "2021-06-02 20:00:00",
                           "2021-06-01 19:00:00", "2021-06-01 21:00:00"],
            "stai_stress": [60, 30, 35, 55],
            "stai_stress_category": ["Above average", "Below average",
                                     "Average", "Above average"],
        }).to_csv(root / "scored_surveys" / "stai.csv", index=False)

        cls.dataset = LifeSnapsDataset(root=str(root))
        cls.samples = cls.dataset.set_task(StressDetectionLifeSnaps())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_dataset_loading(self):
        # p3's only day is incomplete -> p3 has no complete-day events
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_complete_day_filter_and_dedup(self):
        daily = pd.read_csv(Path(self.temp_dir) / "lifesnaps-daily.csv")
        self.assertEqual(len(daily), 4)  # p1 x2, p2 x1, p4 x1
        self.assertNotIn("p3", set(daily["patient_id"]))
        hourly = pd.read_csv(Path(self.temp_dir) / "lifesnaps-hourly.csv")
        p4 = hourly[hourly["patient_id"] == "p4"]
        self.assertEqual(len(p4), 24)  # duplicate hour removed

    def test_task_samples_and_labels(self):
        self.assertEqual(len(self.samples), 3)  # labeled AND complete only
        by_key = {(s["patient_id"], s["day"]): s["stress"]
                  for s in self.samples}
        self.assertEqual(by_key[("p1", "2021-06-01")], 1)
        self.assertEqual(by_key[("p1", "2021-06-02")], 0)
        self.assertEqual(by_key[("p2", "2021-06-01")], 0)

    def test_sensor_shape_and_missing_handling(self):
        s = self.samples[0]
        arr = np.asarray(s["sensors"], dtype=float)
        self.assertEqual(arr.shape, (5, 24))
        self.assertTrue(np.isfinite(arr).all())
        ctx = np.asarray(s["context"], dtype=float)
        self.assertEqual(
            len(ctx), len(CONTEXT_NUMERIC) + len(CONTEXT_CATEGORICAL)
        )

    def test_no_label_leakage_into_inputs(self):
        self.assertFalse(
            set(CONTEXT_NUMERIC + CONTEXT_CATEGORICAL) & EXCLUDED_FEATURES
        )
        daily_cols = set(
            pd.read_csv(Path(self.temp_dir) / "lifesnaps-daily.csv").columns
        )
        self.assertIn("stress_label", daily_cols)  # label exists in table...
        used = set(CONTEXT_NUMERIC + CONTEXT_CATEGORICAL)
        self.assertNotIn("stress_label", used)  # ...but never as input

    def test_task_configurations(self):
        with self.assertRaises(ValueError):
            StressDetectionLifeSnaps(feature_set="bogus")
        sensors_only = self.dataset.set_task(
            StressDetectionLifeSnaps(feature_set="sensors")
        )
        self.assertTrue(
            all(float(v) == 0.0 for s in sensors_only for v in s["context"])
        )
        stale = self.dataset.set_task(
            StressDetectionLifeSnaps(context_staleness_days=1)
        )
        self.assertEqual(len(stale), 3)  # same samples, shifted context
        base = {(s["patient_id"], s["day"]): s for s in self.samples}
        for s in stale:
            self.assertEqual(
                int(s["stress"]), int(base[(s["patient_id"], s["day"])]["stress"])
            )

    def test_model_forward_shapes_and_gradients(self):
        sensors = torch.tensor(
            np.asarray([s["sensors"] for s in self.samples], dtype="float32")
        )
        context = torch.tensor(
            np.asarray([s["context"] for s in self.samples], dtype="float32")
        )
        labels = torch.tensor(
            [int(s["stress"]) for s in self.samples], dtype=torch.long
        )
        model = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(5 * 24 + context.shape[1], 2)
        )
        x = torch.cat([sensors.flatten(1), context], dim=1)
        logits = model[1](x)
        self.assertEqual(logits.shape, (3, 2))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
