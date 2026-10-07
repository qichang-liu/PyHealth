pyhealth.tasks.StressDetectionLifeSnaps
========================================

Binary daily stress detection on the LifeSnaps wearable dataset, following
Yang, Yu & Sano, "Contrastive Pretraining for Stress Detection with
Multimodal Wearable Sensor Data and Surveys" (CHIL 2025,
`PMLR v287 <https://proceedings.mlr.press/v287/yang25a.html>`_).

Each sample is one labeled participant-day with a ``(5, 24)`` hourly sensor
tensor (steps, heart rate, calories, distance, skin temperature) plus a
static/daily context vector (heart-rate zone minutes, demographics, place
categories, IPIP personality). The label is the STAI-derived stress category
(Above average -> stressed). STAI-derived fields are excluded from the inputs
to prevent label leakage.

.. autoclass:: pyhealth.tasks.StressDetectionLifeSnaps
    :members:
    :undoc-members:
    :show-inheritance:
