"""LifeSnaps wearable dataset.

Contributors: Xiangwen Wang (xw120@illinois.edu),
Qichang Liu (qichang4@illinois.edu).
Paper: Yang, Yu & Sano, "Contrastive Pretraining for Stress Detection with
Multimodal Wearable Sensor Data and Surveys", CHIL 2025, PMLR v287.
https://proceedings.mlr.press/v287/yang25a.html
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

_TS_COLS = ["steps", "bpm", "calories", "distance", "temperature"]
_ZONE_COLS = [
    "minutes_in_default_zone_1",
    "minutes_below_default_zone_1",
    "minutes_in_default_zone_2",
    "minutes_in_default_zone_3",
]
_PLACE_COLS = [
    "ENTERTAINMENT",
    "GYM",
    "HOME",
    "HOME_OFFICE",
    "OTHER",
    "OUTDOORS",
    "TRANSIT",
    "WORK/SCHOOL",
]
_STATIC_COLS = ["age", "gender", "bmi", "step_goal_label"]
_PERSONALITY_COLS = [
    "ipip_extraversion_category",
    "ipip_agreeableness_category",
    "ipip_conscientiousness_category",
    "ipip_stability_category",
    "ipip_intellect_category",
]


class LifeSnapsDataset(BaseDataset):
    """LifeSnaps: a 4-month multimodal Fitbit dataset (71 participants).

    LifeSnaps (Yfantidou et al., Scientific Data 2022) contains ~4 months of
    in-the-wild Fitbit Sense data (hourly steps, heart rate, calories,
    distance, skin temperature), surveys (demographics, IPIP personality),
    and scored STAI stress reports for 71 participants. It is distributed
    openly under CC BY 4.0 at https://zenodo.org/records/7229547 (download
    ``rais_anonymized.zip`` and unzip; pass the ``rais_anonymized`` directory
    as ``root``).

    On first use, this class builds two intermediate CSVs inside ``root``:

    - ``lifesnaps-hourly.csv``: one row per participant-day-hour with the five
      sensor channels, restricted to *complete* days (after de-duplicating
      hours, a day must contain exactly the hours 0..23 in order) following
      Yang et al. (CHIL 2025);
    - ``lifesnaps-daily.csv``: one row per complete participant-day with daily
      context (heart-rate zone minutes, demographics, place categories,
      IPIP personality) and, where available, the binary stress label derived
      from the scored STAI category (Above average -> 1; Average or Below
      average -> 0). Missing categories remain unlabeled; unknown nonempty
      categories raise ``ValueError``.

    Citations:
        Yfantidou et al. "LifeSnaps, a 4-month multi-modal dataset capturing
        unobtrusive snapshots of our lives in the wild." Sci Data 9, 663
        (2022). Yang, Yu & Sano. "Contrastive Pretraining for Stress
        Detection with Multimodal Wearable Sensor Data and Surveys."
        CHIL 2025, PMLR 287:166-178.

    Args:
        root: path to the unzipped ``rais_anonymized`` directory.
        dataset_name: optional dataset name, defaults to ``"lifesnaps"``.
        config_path: optional config path, defaults to ``lifesnaps.yaml``.
        cache_dir: optional directory for processed data caches.
        num_workers: worker count. The default one executes event conversion
            synchronously without a distributed scheduler or local server.

    Examples:
        >>> from pyhealth.datasets import LifeSnapsDataset
        >>> dataset = LifeSnapsDataset(root="/path/to/rais_anonymized")
        >>> dataset.stats()
        >>> patient = dataset.get_patient(dataset.unique_patient_ids[0])
        >>> hourly = patient.get_events(event_type="lifesnaps_hourly")
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | Path | None = None,
        num_workers: int = 1,
    ) -> None:
        if config_path is None:
            logger.info("No config provided, using default lifesnaps.yaml")
            config_path = Path(__file__).parent / "configs" / "lifesnaps.yaml"
        if (
            not (Path(root) / "lifesnaps-hourly.csv").exists()
            or not (Path(root) / "lifesnaps-daily.csv").exists()
        ):
            self.prepare_metadata(root)
        super().__init__(
            root=root,
            tables=["lifesnaps_hourly", "lifesnaps_daily"],
            dataset_name=dataset_name or "lifesnaps",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
        )

    @staticmethod
    def _mode_or_nan(series: pd.Series) -> Any:
        """Return the most common nonmissing scalar, or NaN for an empty column."""
        series = series.dropna()
        return series.mode().iloc[0] if len(series) else np.nan

    def prepare_metadata(self, root: str) -> None:
        """Builds the two intermediate CSVs from the raw Zenodo layout.

        Keeps only *complete* participant-days: after de-duplicating the
        ``hour`` column, a day is kept iff it has exactly 24 rows whose hour
        values appear in order 0..23 (the inclusion rule of Yang et al.,
        verified to reproduce the paper's 5850 days / 228 labeled days).
        Sensor gaps inside a kept day are linearly interpolated then
        zero-filled. The STAI-derived stress label is stored ONLY in
        ``stress_label`` so tasks can exclude it from model inputs.

        Args:
            root: path to the unzipped ``rais_anonymized`` directory.

        Raises:
            ValueError: If no complete participant-days remain, or a nonempty
                STAI category is not Above average, Average, or Below average.
        """
        root_path = Path(root)
        hourly_csv = (
            root_path / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv"
        )
        usecols = (
            ["id", "date", "hour"]
            + _TS_COLS
            + _ZONE_COLS
            + _PLACE_COLS
            + _STATIC_COLS
            + ["activityType"]
        )
        hourly = pd.read_csv(hourly_csv, usecols=usecols, low_memory=False)
        hourly["day"] = pd.to_datetime(hourly["date"]).dt.date.astype(str)

        hourly_rows: list[pd.DataFrame] = []
        daily_rows: list[dict] = []
        for (pid, day), group in hourly.groupby(["id", "day"], sort=True):
            group = group.drop_duplicates(subset="hour")
            if group.shape[0] != 24:
                continue
            if not (group["hour"].values == np.arange(24)).all():
                continue
            sensors = group[_TS_COLS].interpolate().fillna(0)
            frame = pd.DataFrame(
                {
                    "patient_id": pid,
                    "timestamp": [f"{day} {h:02d}:00:00" for h in range(24)],
                    "day": day,
                    "hour": np.arange(24),
                }
            )
            hourly_rows.append(
                pd.concat([frame, sensors.reset_index(drop=True)], axis=1)
            )
            row = {"patient_id": pid, "day": day}
            for col in _ZONE_COLS:
                row[col.lower()] = group[col].sum()
            for col in _STATIC_COLS + _PLACE_COLS + ["activityType"]:
                key = (
                    col.lower()
                    .replace("/", "_")
                    .replace("activitytype", "activity_type")
                )
                row[key] = self._mode_or_nan(group[col])
            daily_rows.append(row)

        if not daily_rows:
            raise ValueError(
                "LifeSnaps has no complete participant-days (hours 0..23)."
            )
        daily = pd.DataFrame(daily_rows)

        personality = pd.read_csv(
            root_path / "scored_surveys" / "personality.csv"
        ).rename(columns={"user_id": "patient_id"})
        personality = personality[["patient_id"] + _PERSONALITY_COLS].drop_duplicates(
            "patient_id"
        )
        daily = daily.merge(personality, on="patient_id", how="left")

        stai = pd.read_csv(root_path / "scored_surveys" / "stai.csv").rename(
            columns={"user_id": "patient_id"}
        )
        stai["day"] = pd.to_datetime(stai["submitdate"]).dt.date.astype(str)
        category = stai["stai_stress_category"].astype("string").str.strip()
        category = category.replace("", pd.NA)
        label_map = {"Above average": 1, "Average": 0, "Below average": 0}
        invalid = category.notna() & ~category.isin(label_map)
        if invalid.any():
            unknown = sorted(category[invalid].unique().tolist())
            raise ValueError(f"Unknown STAI stress categories: {unknown}")
        stai["stress_label"] = category.map(label_map)
        stai = stai[["patient_id", "day", "stress_label"]].drop_duplicates(
            ["patient_id", "day"]
        )
        daily = daily.merge(stai, on=["patient_id", "day"], how="left")

        pd.concat(hourly_rows).to_csv(root_path / "lifesnaps-hourly.csv", index=False)
        daily.to_csv(root_path / "lifesnaps-daily.csv", index=False)
        n_labeled = int(daily["stress_label"].notna().sum())
        logger.info(
            "LifeSnaps prepared: %d complete days, %d labeled",
            len(daily),
            n_labeled,
        )
