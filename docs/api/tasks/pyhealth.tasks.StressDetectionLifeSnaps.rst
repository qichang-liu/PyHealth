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

``feature_set`` can retain both inputs or zero one modality. The historical
``context_staleness_days`` parameter counts previous available observations,
not calendar days; insufficient history uses the first available record.
``context_text`` is additional sample metadata describing only allowlisted
raw context fields from the same selected record. It is not part of the
numeric model input schema, and contains no STAI-derived fields.

For contrastive pretraining, ``StressDetectionLifeSnaps(pretraining=True)``
includes all complete days, including days without stress labels. This mode
has an empty output schema and does not emit a ``stress`` field. The default
mode remains labeled binary classification. Callers must split participants
before pretraining so that validation/test people do not enter its data.

.. code-block:: python

    labeled = dataset.set_task(StressDetectionLifeSnaps())
    unlabeled_view = dataset.set_task(StressDetectionLifeSnaps(pretraining=True))
    # Select train participants from unlabeled_view before contrastive training.

.. autoclass:: pyhealth.tasks.StressDetectionLifeSnaps
    :members:
    :undoc-members:
    :show-inheritance:
