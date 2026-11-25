"""Daily stress detection task on LifeSnaps.

Contributors: Xiangwen Wang (xw120@illinois.edu),
Qichang Liu (qichang4@illinois.edu).
Paper: Yang, Yu & Sano, "Contrastive Pretraining for Stress Detection with
Multimodal Wearable Sensor Data and Surveys", CHIL 2025, PMLR v287.
https://proceedings.mlr.press/v287/yang25a.html
"""

import hashlib
from typing import Any, ClassVar

import numpy as np

from pyhealth.tasks import BaseTask

# STAI/label-derived fields must never enter model inputs (leakage guard).
EXCLUDED_FEATURES = frozenset({"stress_label", "stai_stress", "stai_stress_category"})

CONTEXT_NUMERIC = [
    "minutes_in_default_zone_1",
    "minutes_below_default_zone_1",
    "minutes_in_default_zone_2",
    "minutes_in_default_zone_3",
]
CONTEXT_CATEGORICAL = [
    "age",
    "gender",
    "bmi",
    "step_goal_label",
    "entertainment",
    "gym",
    "home",
    "home_office",
    "other",
    "outdoors",
    "transit",
    "work_school",
    "activity_type",
    "ipip_extraversion_category",
    "ipip_agreeableness_category",
    "ipip_conscientiousness_category",
    "ipip_stability_category",
    "ipip_intellect_category",
]
SENSOR_CHANNELS = ["steps", "bpm", "calories", "distance", "temperature"]


class StressDetectionLifeSnaps(BaseTask):
    """Binary daily stress detection from wearable sensors + survey context.

    Following Yang et al. (CHIL 2025), each sample is one labeled
    participant-day: the inputs are the day's hourly sensor tensor with shape
    ``(5, 24)`` (steps, heart rate, calories, distance, skin temperature) and
    a static/daily context vector (heart-rate zone minutes, demographics,
    place categories, IPIP personality; categoricals are numerically encoded,
    missing values are encoded as ``-1``). String categories use a stable
    feature-scoped hash; this is not identical to the reference code's fitted
    ordinal encoding. The output is the
    binary STAI-derived stress label (Above average -> 1; Average or Below
    average -> 0). With ``pretraining=True``, all complete days are included
    and neither a stress field nor an output label schema is produced.

    STAI-derived fields are excluded from the inputs via
    ``EXCLUDED_FEATURES`` to prevent label leakage.

    Attributes:
        task_name (str): "StressDetectionLifeSnaps".
        input_schema (Dict[str, str]): {"sensors": "tensor",
            "context": "tensor"}.
        output_schema (Dict[str, str]): {"stress": "binary"}.

    Examples:
        >>> from pyhealth.datasets import LifeSnapsDataset
        >>> from pyhealth.tasks import StressDetectionLifeSnaps
        >>> dataset = LifeSnapsDataset(root="/path/to/rais_anonymized")
        >>> samples = dataset.set_task(StressDetectionLifeSnaps())
        >>> samples[0]["sensors"].shape
        torch.Size([5, 24])
    """

    task_name: str = "StressDetectionLifeSnaps"
    input_schema: ClassVar[dict[str, str]] = {
        "sensors": "tensor",
        "context": "tensor",
    }
    output_schema: ClassVar[dict[str, str]] = {"stress": "binary"}

    def __init__(
        self,
        feature_set: str = "both",
        context_staleness_days: int = 0,
        pretraining: bool = False,
    ) -> None:
        """Configures the task's feature variations (for ablation studies).

        Args:
            feature_set: which modalities feed the model — "both" (default),
                "sensors" (context zeroed out), or "context" (sensors zeroed
                out).
            context_staleness_days: number of available daily observations
                to shift back (0 = same-day). Despite the historical name,
                this is not a calendar-day interval when records have gaps.
                Insufficient history falls back to the first observation,
                which can be more recent than the requested offset.
            pretraining: If True, include complete unlabeled days and omit
                all target labels. Split by participant before using these
                samples to ensure validation/test people do not enter pretraining.

        Raises:
            ValueError: If the feature set or observation offset is invalid.
        """
        if feature_set not in {"both", "sensors", "context"}:
            raise ValueError(f"invalid feature_set: {feature_set}")
        if (
            not isinstance(context_staleness_days, int)
            or isinstance(context_staleness_days, bool)
            or context_staleness_days < 0
        ):
            raise ValueError("context_staleness_days must be a nonnegative integer")
        self.feature_set = feature_set
        self.context_staleness_days = context_staleness_days
        self.pretraining = pretraining
        if pretraining:
            self.output_schema = {}
        super().__init__()

    @staticmethod
    def _to_float(value: Any) -> float:
        """Casts an event attribute to float; missing/non-numeric -> -1."""
        if value is None:
            return -1.0
        try:
            out = float(value)
        except (TypeError, ValueError):
            return -1.0
        return -1.0 if np.isnan(out) else out

    def _encode_context(self, event: Any) -> list[float]:
        """Encode only allowlisted fields, using -1 for missing values."""
        values: list[float] = []
        for key in CONTEXT_NUMERIC:
            values.append(self._to_float(getattr(event, key, None)))
        for key in CONTEXT_CATEGORICAL:
            raw = getattr(event, key, None)
            missing = (
                raw is None
                or (isinstance(raw, float) and np.isnan(raw))
                or (isinstance(raw, str) and raw.strip() == "")
            )
            if missing:
                values.append(-1.0)
            else:
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    # Stable across processes; include the column name so
                    # different fields do not share a raw-string token.
                    token = f"{key}:{raw}".encode()
                    code = (
                        int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % 1000
                    )
                    values.append(float(code))
        return values

    @staticmethod
    def _context_text(event: Any) -> str:
        """Describe allowlisted raw context fields without stress labels."""
        fields = []
        for key in CONTEXT_NUMERIC + CONTEXT_CATEGORICAL:
            raw = getattr(event, key, None)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            value = str(raw).strip()
            if value:
                fields.append(f"{key.replace('_', ' ')}: {value}")
        return ". ".join(fields)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build samples from complete days, including unlabeled pretraining days.

        Args:
            patient: a PyHealth patient with ``lifesnaps_hourly`` and
                ``lifesnaps_daily`` events.

        Returns:
            A list of sample dicts with keys ``patient_id``, ``day``,
            ``sensors`` ((5, 24) list), ``context`` (list of floats), and
            ``context_text`` (raw context description, not a model input key).
            Supervised samples additionally contain ``stress`` (0/1 int).

        Raises:
            ValueError: If a nonmissing supervised label is not binary.
        """
        daily = patient.get_events(event_type="lifesnaps_daily")
        hourly = patient.get_events(event_type="lifesnaps_hourly")
        by_day: dict[str, list] = {}
        for ev in hourly:
            by_day.setdefault(str(ev.day), []).append(ev)
        daily_sorted = sorted(daily, key=lambda e: str(e.day))
        day_index = {str(e.day): i for i, e in enumerate(daily_sorted)}

        samples: list[dict[str, Any]] = []
        for ev in daily:
            label_f = -1.0
            if not self.pretraining:
                label = getattr(ev, "stress_label", None)
                if label is None or (isinstance(label, str) and label.strip() == ""):
                    continue
                label_f = self._to_float(label)
                if label_f < 0:
                    continue
                if label_f not in (0.0, 1.0):
                    raise ValueError(f"Expected a binary stress label, got {label!r}")
            day = str(ev.day)
            hours = sorted(by_day.get(day, []), key=lambda e: int(float(e.hour)))
            if len(hours) != 24 or [int(float(h.hour)) for h in hours] != list(
                range(24)
            ):
                continue
            sensors = [
                [self._to_float(getattr(h, ch, None)) for h in hours]
                for ch in SENSOR_CHANNELS
            ]
            # leakage guard: context never contains STAI-derived fields
            assert not (set(CONTEXT_NUMERIC + CONTEXT_CATEGORICAL) & EXCLUDED_FEATURES)
            context_event = ev
            if self.context_staleness_days > 0:
                pos = day_index[day] - self.context_staleness_days
                context_event = daily_sorted[max(0, pos)]
            context = self._encode_context(context_event)
            if self.feature_set == "sensors":
                context = [0.0] * len(context)
            elif self.feature_set == "context":
                sensors = [[0.0] * 24 for _ in SENSOR_CHANNELS]
            sample = {
                "patient_id": patient.patient_id,
                "day": day,
                "sensors": sensors,
                "context": context,
                "context_text": ""
                if self.feature_set == "sensors"
                else self._context_text(context_event),
            }
            if not self.pretraining:
                sample["stress"] = int(label_f)
            samples.append(sample)
        return samples
