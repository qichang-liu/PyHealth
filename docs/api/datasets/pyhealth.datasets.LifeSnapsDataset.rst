pyhealth.datasets.LifeSnapsDataset
===================================

LifeSnaps (Yfantidou et al., Scientific Data 2022) is a 4-month, in-the-wild
multimodal dataset from 71 participants wearing Fitbit Sense smartwatches:
hourly sensor streams (steps, heart rate, calories, distance, skin
temperature), surveys (demographics, IPIP-50 personality), and scored STAI
stress reports. It is openly distributed under CC BY 4.0.

Download ``rais_anonymized.zip`` from `Zenodo
<https://zenodo.org/records/7229547>`_, unzip it, and pass the
``rais_anonymized`` directory as ``root``. On first use the loader keeps only
*complete* participant-days (exactly hours 0..23 after de-duplication),
following Yang, Yu & Sano (CHIL 2025,
`PMLR v287 <https://proceedings.mlr.press/v287/yang25a.html>`_), and derives
the binary daily stress label from the scored STAI category. Label-derived
fields are kept out of model inputs by the companion task
:class:`pyhealth.tasks.StressDetectionLifeSnaps`.

.. autoclass:: pyhealth.datasets.LifeSnapsDataset
    :members:
    :undoc-members:
    :show-inheritance:
