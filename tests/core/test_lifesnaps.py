"""Tests for LifeSnapsDataset and StressDetectionLifeSnaps (synthetic data).

Contributors: Xiangwen Wang (xw120@illinois.edu),
Qichang Liu (qichang4@illinois.edu).
Paper: https://proceedings.mlr.press/v287/yang25a.html
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pyhealth.datasets import LifeSnapsDataset
from pyhealth.tasks import StressDetectionLifeSnaps
from pyhealth.tasks.stress_detection_lifesnaps import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    EXCLUDED_FEATURES,
)

PLACE = [
    "ENTERTAINMENT",
    "GYM",
    "HOME",
    "HOME_OFFICE",
    "OTHER",
    "OUTDOORS",
    "TRANSIT",
    "WORK/SCHOOL",
]


class TestLifeSnaps(unittest.TestCase):
    """Synthetic 4-participant fixture exercising dataset + task."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.temp_dir)
        root = Path(cls.temp_dir)
        (root / "csv_rais_anonymized").mkdir()
        (root / "scored_surveys").mkdir()
        rng = np.random.default_rng(0)

        rows = []
        # p1: 2 complete days; p2: 1 complete day; p3: INCOMPLETE day (23h);
        # p4: complete day with missing sensor values and a duplicate hour.
        plan = {
            "p1": [("2021-06-01", 24), ("2021-06-02", 24)],
            "p2": [("2021-06-01", 24)],
            "p3": [("2021-06-01", 23)],
            "p4": [("2021-06-03", 24)],
        }
        for pid, days in plan.items():
            for day, n_hours in days:
                for h in range(n_hours):
                    rows.append(
                        {
                            "id": pid,
                            "date": day,
                            "hour": h,
                            "steps": np.nan
                            if (pid == "p4" and h < 3)
                            else float(rng.integers(0, 500)),
                            "bpm": float(rng.integers(55, 110)),
                            "calories": float(rng.integers(50, 300)),
                            "distance": float(rng.random()),
                            "temperature": 33.0 + rng.random(),
                            "minutes_in_default_zone_1": 9
                            if pid == "p1" and day == "2021-06-02"
                            else 5,
                            "minutes_below_default_zone_1": 40,
                            "minutes_in_default_zone_2": 2,
                            "minutes_in_default_zone_3": 0,
                            "age": "<30",
                            "gender": "MALE",
                            "bmi": "<25",
                            "step_goal_label": "below goal",
                            **{c: 0 for c in PLACE},
                            "activityType": np.nan,
                        }
                    )
        # duplicate-hour row for p4 (dataset must de-duplicate, keep 24)
        rows.append(dict(rows[-1], hour=23))
        pd.DataFrame(rows).to_csv(
            root / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv",
            index=False,
        )

        pd.DataFrame(
            {
                "user_id": ["p1", "p2", "p3", "p4"],
                "ipip_extraversion_category": ["High", "Low", "Average", "High"],
                "ipip_agreeableness_category": ["Average"] * 4,
                "ipip_conscientiousness_category": ["Low"] * 4,
                "ipip_stability_category": ["High"] * 4,
                "ipip_intellect_category": ["Average"] * 4,
            }
        ).to_csv(root / "scored_surveys" / "personality.csv", index=False)

        # labels: p1 day1 stressed, p1 day2 relaxed, p2 day1 relaxed,
        # p3 labeled but its day is incomplete -> must NOT become a sample;
        # p4 unlabeled.
        pd.DataFrame(
            {
                "user_id": ["p1", "p1", "p2", "p3"],
                "submitdate": [
                    "2021-06-01 20:00:00",
                    "2021-06-02 20:00:00",
                    "2021-06-01 19:00:00",
                    "2021-06-01 21:00:00",
                ],
                "stai_stress": [60, 30, 35, 55],
                "stai_stress_category": [
                    "Above average",
                    "Below average",
                    "Average",
                    "Above average",
                ],
            }
        ).to_csv(root / "scored_surveys" / "stai.csv", index=False)

        cls.dataset = LifeSnapsDataset(root=str(root), cache_dir=root / "cache")
        cls.samples = cls.dataset.set_task(StressDetectionLifeSnaps())

    def test_dataset_loading(self):
        # p3's only day is incomplete -> p3 has no complete-day events
        self.assertEqual(set(self.dataset.unique_patient_ids), {"p1", "p2", "p4"})
        patient = self.dataset.get_patient("p1")
        self.assertEqual(patient.patient_id, "p1")
        hourly = patient.get_events(event_type="lifesnaps_hourly")
        daily = patient.get_events(event_type="lifesnaps_daily")
        self.assertEqual(len(hourly), 48)
        self.assertEqual(len(daily), 2)
        self.assertEqual(hourly[0].timestamp.isoformat(), "2021-06-01T00:00:00")
        day1 = [event for event in hourly if event.day == "2021-06-01"]
        self.assertEqual([int(event.hour) for event in day1], list(range(24)))
        by_day = {event.day: event for event in daily}
        self.assertEqual(float(by_day["2021-06-01"].minutes_in_default_zone_1), 120)
        self.assertEqual(float(by_day["2021-06-02"].minutes_in_default_zone_1), 216)
        self.assertEqual(by_day["2021-06-01"].gender, "MALE")
        self.assertEqual(float(by_day["2021-06-01"].stress_label), 1)

    def test_complete_day_filter_and_dedup(self):
        daily = pd.read_csv(Path(self.temp_dir) / "lifesnaps-daily.csv")
        self.assertEqual(len(daily), 4)  # p1 x2, p2 x1, p4 x1
        self.assertNotIn("p3", set(daily["patient_id"]))
        hourly = pd.read_csv(Path(self.temp_dir) / "lifesnaps-hourly.csv")
        p4 = hourly[hourly["patient_id"] == "p4"]
        self.assertEqual(len(p4), 24)  # duplicate hour removed

    def test_task_samples_and_labels(self):
        self.assertEqual(len(self.samples), 3)  # labeled AND complete only
        by_key = {(s["patient_id"], s["day"]): s["stress"] for s in self.samples}
        self.assertEqual(by_key[("p1", "2021-06-01")], 1)
        self.assertEqual(by_key[("p1", "2021-06-02")], 0)
        self.assertEqual(by_key[("p2", "2021-06-01")], 0)

    def test_sensor_shape_and_missing_handling(self):
        s = self.samples[0]
        arr = np.asarray(s["sensors"], dtype=float)
        self.assertEqual(arr.shape, (5, 24))
        self.assertTrue(np.isfinite(arr).all())
        ctx = np.asarray(s["context"], dtype=float)
        self.assertEqual(len(ctx), len(CONTEXT_NUMERIC) + len(CONTEXT_CATEGORICAL))

        # p4 is unlabeled, so inspect its prepared events rather than a
        # labeled task sample to exercise the fixture's missing step values.
        hourly = pd.read_csv(Path(self.temp_dir) / "lifesnaps-hourly.csv")
        p4 = hourly[hourly["patient_id"] == "p4"].sort_values("hour")
        self.assertTrue(np.isfinite(p4["steps"].to_numpy()).all())
        np.testing.assert_array_equal(p4["steps"].to_numpy()[:3], [0, 0, 0])

    def test_no_label_leakage_into_inputs(self):
        self.assertFalse(set(CONTEXT_NUMERIC + CONTEXT_CATEGORICAL) & EXCLUDED_FEATURES)
        daily_cols = set(
            pd.read_csv(Path(self.temp_dir) / "lifesnaps-daily.csv").columns
        )
        self.assertIn("stress_label", daily_cols)  # label exists in table...
        used = set(CONTEXT_NUMERIC + CONTEXT_CATEGORICAL)
        self.assertNotIn("stress_label", used)  # ...but never as input
        for sample in self.samples:
            self.assertNotIn("stai", sample["context_text"].lower())
            self.assertNotIn("stress", sample["context_text"].lower())

    def test_task_configurations(self):
        with self.assertRaises(ValueError):
            StressDetectionLifeSnaps(feature_set="bogus")
        for invalid_offset in (-1, 0.5, True):
            with (
                self.subTest(invalid_offset=invalid_offset),
                self.assertRaises(ValueError),
            ):
                StressDetectionLifeSnaps(context_staleness_days=invalid_offset)
        sensors_only = self.dataset.set_task(
            StressDetectionLifeSnaps(feature_set="sensors")
        )
        self.assertTrue(
            all(float(v) == 0.0 for s in sensors_only for v in s["context"])
        )
        self.assertTrue(all(s["context_text"] == "" for s in sensors_only))
        context_only = self.dataset.set_task(
            StressDetectionLifeSnaps(feature_set="context")
        )
        self.assertTrue(
            all(np.count_nonzero(np.asarray(s["sensors"])) == 0 for s in context_only)
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
            np.testing.assert_array_equal(
                s["sensors"], base[(s["patient_id"], s["day"])]["sensors"]
            )
        by_key = {(s["patient_id"], s["day"]): s for s in stale}
        current = base[("p1", "2021-06-02")]
        previous = base[("p1", "2021-06-01")]
        delayed = by_key[("p1", "2021-06-02")]
        self.assertEqual(float(current["context"][0]), 216)
        self.assertEqual(float(delayed["context"][0]), 120)
        np.testing.assert_array_equal(delayed["context"], previous["context"])
        self.assertEqual(delayed["context_text"], previous["context_text"])
        np.testing.assert_array_equal(
            by_key[("p1", "2021-06-01")]["context"], previous["context"]
        )

    def test_pretraining_includes_unlabeled_without_labels(self):
        task = StressDetectionLifeSnaps(pretraining=True)
        self.assertEqual(task.output_schema, {})
        self.assertEqual(StressDetectionLifeSnaps().output_schema, {"stress": "binary"})
        samples = self.dataset.set_task(task)
        self.assertEqual(len(samples), 4)
        self.assertEqual({s["patient_id"] for s in samples}, {"p1", "p2", "p4"})
        for sample in samples:
            self.assertNotIn("stress", sample)
            self.assertEqual(tuple(sample["sensors"].shape), (5, 24))
            self.assertIn("gender: MALE", sample["context_text"])
            self.assertNotIn("stai", sample["context_text"].lower())

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


@pytest.fixture
def metadata_root(tmp_path: Path) -> Path:
    """Two synthetic people with three complete days each; no Dask needed."""
    (tmp_path / "csv_rais_anonymized").mkdir()
    (tmp_path / "scored_surveys").mkdir()
    rows = []
    labels = []
    for pid in ("p1", "p2"):
        for day_number, category in enumerate(
            ("Above average", "Average", "Below average"), start=1
        ):
            day = f"2021-06-{day_number:02d}"
            labels.append(
                {
                    "user_id": pid,
                    "submitdate": f"{day} 20:00:00",
                    "stai_stress_category": category,
                }
            )
            for hour in range(24):
                rows.append(
                    {
                        "id": pid,
                        "date": day,
                        "hour": hour,
                        "steps": 100,
                        "bpm": 70,
                        "calories": 50,
                        "distance": 1,
                        "temperature": 33,
                        "minutes_in_default_zone_1": 5,
                        "minutes_below_default_zone_1": 40,
                        "minutes_in_default_zone_2": 2,
                        "minutes_in_default_zone_3": 0,
                        "age": "<30",
                        "gender": "MALE",
                        "bmi": "<25",
                        "step_goal_label": "below goal",
                        **{place: 0 for place in PLACE},
                        "activityType": np.nan,
                    }
                )
    pd.DataFrame(rows).to_csv(
        tmp_path / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "user_id": ["p1", "p2"],
            **{
                field: ["Average", "Average"]
                for field in (
                    "ipip_extraversion_category",
                    "ipip_agreeableness_category",
                    "ipip_conscientiousness_category",
                    "ipip_stability_category",
                    "ipip_intellect_category",
                )
            },
        }
    ).to_csv(tmp_path / "scored_surveys" / "personality.csv", index=False)
    pd.DataFrame(labels).to_csv(tmp_path / "scored_surveys" / "stai.csv", index=False)
    return tmp_path


def test_metadata_labels_keep_missing_unlabeled(metadata_root: Path) -> None:
    """A recorded survey with an absent category must not become a negative."""
    path = metadata_root / "scored_surveys" / "stai.csv"
    stai = pd.read_csv(path)
    stai.loc[stai["user_id"] == "p2", "stai_stress_category"] = [np.nan, "", " "]
    stai.to_csv(path, index=False)

    # Exercise raw-file preparation directly, without BaseDataset's Dask layer.
    dataset = LifeSnapsDataset.__new__(LifeSnapsDataset)
    dataset.prepare_metadata(str(metadata_root))
    daily = pd.read_csv(metadata_root / "lifesnaps-daily.csv")
    assert len(daily) == 6
    labels = daily.set_index(["patient_id", "day"])["stress_label"]
    assert labels.loc[("p1", "2021-06-01")] == 1
    assert labels.loc[("p1", "2021-06-02")] == 0
    assert labels.loc[("p1", "2021-06-03")] == 0
    assert labels.loc["p2"].isna().all()
    assert labels.notna().sum() == 3


def test_metadata_rejects_unknown_category(metadata_root: Path) -> None:
    path = metadata_root / "scored_surveys" / "stai.csv"
    stai = pd.read_csv(path)
    stai.loc[0, "stai_stress_category"] = "Unexpected category"
    stai.to_csv(path, index=False)

    dataset = LifeSnapsDataset.__new__(LifeSnapsDataset)
    with pytest.raises(ValueError, match="Unknown STAI stress categories"):
        dataset.prepare_metadata(str(metadata_root))
    assert not (metadata_root / "lifesnaps-daily.csv").exists()
    assert not (metadata_root / "lifesnaps-hourly.csv").exists()


def test_metadata_rejects_no_complete_days(metadata_root: Path) -> None:
    path = (
        metadata_root / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv"
    )
    hourly = pd.read_csv(path)
    hourly[hourly["hour"] < 23].to_csv(path, index=False)

    dataset = LifeSnapsDataset.__new__(LifeSnapsDataset)
    with pytest.raises(ValueError, match="no complete participant-days"):
        dataset.prepare_metadata(str(metadata_root))


def test_fixture_cleanup_after_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unittest class cleanup must run even when dataset construction fails."""

    class FailingSetup(TestLifeSnaps):
        pass

    def fail_dataset_construction(*args, **kwargs):
        raise RuntimeError("deliberate fixture setup failure")

    monkeypatch.setitem(globals(), "LifeSnapsDataset", fail_dataset_construction)
    suite = unittest.TestSuite([FailingSetup("test_dataset_loading")])
    result = unittest.TestResult()
    suite.run(result)
    assert len(result.errors) == 1
    assert "deliberate fixture setup failure" in result.errors[0][1]
    assert not Path(FailingSetup.temp_dir).exists()


if __name__ == "__main__":
    unittest.main()
