pyhealth.models.WearableStressModel
===================================

PyHealth implementation of the wearable and context architecture described in
`Yang et al., CHIL 2025
<https://proceedings.mlr.press/v287/yang25a.html>`_, informed by the reference
repository's documented behavior. The author source is not imported.
Sensor channels are Transformer tokens, each containing 24 hourly values.
A residual convolutional branch encodes tabular context. Late fusion is
projected for classification or paired-text contrastive pretraining.

Training stages
---------------

* ``supervised``: optimize a binary classifier from scratch.
* ``pretrain``: consume external 768-dimensional ``text_emb`` vectors and
  optimize a stop-gradient soft-target symmetric contrastive objective.
  This stage returns ``loss`` and ``embed``; it needs a dedicated pretraining
  loop rather than a classification-metrics Trainer call.
* ``probe``: update the classification head only. The representation's
  parameters, dropout and BatchNorm running statistics are frozen.
* ``finetune``: unfreeze and jointly adapt the representation and classifier.

``set_stage`` preserves existing weights. ``save_checkpoint`` and
``load_checkpoint`` preserve the model configuration, active stage and weights.
For classification, forward returns ``logit`` and ``y_prob``; when ``stress``
is supplied it also returns ``loss`` and ``y_true``. The binary output shape
is ``(batch, 1)``. No BERT download occurs in the model or its unit tests.

Complete example and temporal extension
---------------------------------------

``examples/wearable/lifesnaps_stressdetection_wearable_stress.py`` demonstrates
Dataset -> Task -> paired frozen-BERT vectors -> pretraining -> frozen probe
-> fine-tuning -> evaluation and checkpoint restoration. The companion
``lifesnaps_clsp_utils.py`` constructs text from an allowlist of raw context
and sensor statistics. BERT is frozen, and text vectors are cached with a
hash of their text/model configuration. The default model lookup is local;
``--allow-download`` explicitly enables downloading weights.

The example splits participants once, then fits preprocessing and pretrains
only with training participants. ``StressDetectionLifeSnaps(pretraining=True)``
provides complete days without labels. Validation is used for classifier
checkpoint selection; test labels do not select configurations.

The fixed-weight ablation shifts only time-varying context to the latest
same-participant date no later than ``t-k`` for calendar delays 0, 1, 3 and 7.
Static attributes, sensors, labels and test rows stay fixed. If history is
insufficient, a predeclared vector of training-partition coordinate medians
is used for dynamic fields, and fallback counts are reported. The existing
Task parameter with a ``days`` suffix remains an observation offset; this
example's calendar-day intervention is a separate explicit helper.

Use ``--synthetic --output /tmp/lifesnaps-smoke`` for a fast offline software
check with deterministic random text vectors. Such vectors and scores are
not scientific evidence about BERT. For real data use ``--root`` and an
explicit output directory. Calibrate with one epoch per stage before a larger
run; ``--max-pretrain-days`` permits a seeded training-only subset when compute
is limited. Exact commands, settings and measured results are documented in
``examples/wearable/lifesnaps_clsp_results.md`` after validation.

Differences from the publication
-------------------------------

The complete example uses participant-level 60/20/20 splits and an explicit
limited default epoch budget, rather than the random-day Table 1 protocol.
The model has one binary logit; the task uses deterministic category codes;
example quantile transforms include context coordinates; prompts put context
first and report 512-token truncation. These choices are documented adaptation
differences, not a claim of numerical identity with the author's implementation.
The soft-target forward formula is informed by the reference code, but this
implementation detaches target probabilities. The reference allows gradients
through those targets, while the paper states a one-hot objective. These are
distinct optimization objectives; this example is an explicit stop-gradient
adaptation and does not establish equivalence. A high or low AUC alone does not validate the claimed
pretraining mechanism or establish medical utility.

.. autoclass:: pyhealth.models.WearableStressModel
    :members:
    :show-inheritance:
