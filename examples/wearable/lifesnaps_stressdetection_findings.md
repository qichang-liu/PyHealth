# LifeSnaps MLP feature and context-offset pilot

Recorded 2026-11-25. These are actual runs of the **existing PyHealth MLP** using
LifeSnapsDataset and StressDetectionLifeSnaps. They are not the paper's CLSP
results and are not copied from the separate research reproduction pipeline.

Paper: Yang, Yu, and Sano, [Contrastive Pretraining for Stress Detection with
Multimodal Wearable Sensor Data and Surveys](https://proceedings.mlr.press/v287/yang25a.html), CHIL 2025.

## Reproduce the experiment

From the PyHealth repository root, with its dependencies installed:

```sh
python examples/wearable/lifesnaps_stressdetection_mlp.py --synthetic
python examples/wearable/lifesnaps_stressdetection_mlp.py \
  --root /path/to/rais_anonymized --configs both --epochs 1 --seeds 0 \
  --output /tmp/lifesnaps_mlp_calibration.json
python examples/wearable/lifesnaps_stressdetection_mlp.py \
  --root /path/to/rais_anonymized --epochs 20 --seeds 0 1 2 \
  --output /tmp/lifesnaps_mlp_results.json
```

The first command creates four artificial participants and executes all five
configurations; its metrics have no scientific interpretation. No dataset or
checkpoint download occurs during the synthetic demonstration. Temporary raw
data, caches, and training checkpoints are cleaned on exit. Download the real
LifeSnaps release separately from Zenodo record 7229547 for the other commands.

The real pilot used the locally prepared 5,850 complete participant-days from
71 participants, with 228 labeled days belonging to **50 participants**. Existing
prepared CSVs were reused by the loader; the pipeline then built fresh event and
task caches through the actual BaseDataset/Task path. Raw and prepared file
hashes are recorded in the accompanying result JSON. The source STAI file has
279 records, all with recognized nonmissing category values. The software tests
separately exercise missing and malformed STAI categories.

## Prespecified comparison

- Five configurations: both modalities; sensors only; context only; context from
  one earlier observation; context from seven earlier observations.
- A participant-level PatientSplit with ratios 0.6/0.2/0.2 for each seed 0, 1, 2.
  The same split and sample keys are asserted across all five configurations.
  Each participant appears in exactly one partition. Split choice is not based
  on test performance.
- Each seed determines split, initialization and minibatch shuffle. Changing seed
  therefore changes both subjects and random optimization; three seeds are a
  small pilot, not independent clinical validation cohorts.
- Both modalities remain in the model schema; the ablated modality is zeroed.
  This preserves the model's dimensions and parameter count across feature
  configurations.
- Per-coordinate mean and population standard deviation are fitted on the
  training samples of each configuration. Constant coordinates use scale 1.
  Validation and test values cannot affect these fitted statistics.
- MLP embedding dimension 32, hidden dimension 32, two layers per feature, Adam
  learning rate 0.001, batch size 64, 20 epochs, CPU, one PyTorch thread.
- The minimum validation binary-cross-entropy loss selects a checkpoint. The
  test partition is evaluated after this selection, not used to select epochs
  or configurations. Validation-loss selection works even if a partition has one
  class. Test ROC-AUC is explicitly null if the test set has only one class.
- Report probability ROC-AUC, not ROC-AUC of hard predictions. All three real
  test partitions in this pilot contain both classes.

| Seed | Train patients / samples / positives | Validation patients / samples / positives | Test patients / samples / positives |
|---|---|---|---|
| 0 | 30 / 145 / 52 | 10 / 33 / 11 | 10 / 50 / 17 |
| 1 | 30 / 140 / 51 | 10 / 45 / 17 | 10 / 43 / 12 |
| 2 | 30 / 146 / 53 | 10 / 41 / 19 | 10 / 41 / 8 |

## Actual results

Probability ROC-AUC in percent. SD below is the population SD across the three
seed results (ddof=0), **not** a confidence interval or a participant-bootstrap
interval. Delta is the mean within-seed difference from both modalities.

| Configuration | Seed 0 | Seed 1 | Seed 2 | Mean +/- SD | Mean delta (percentage points) |
|---|---|---|---|---|---|
| both | 64.53 | 71.24 | 72.73 | 69.50 +/- 3.57 | +0.00 |
| sensors | 65.95 | 68.55 | 70.08 | 68.19 +/- 1.70 | -1.30 |
| context | 61.76 | 30.91 | 73.86 | 55.51 +/- 18.08 | -13.98 |
| offset1 | 69.25 | 68.28 | 70.45 | 69.33 +/- 0.89 | -0.17 |
| offset7 | 68.36 | 70.16 | 73.86 | 70.79 +/- 2.29 | +1.30 |

Machine-readable evidence: [per-run results and source/data hashes](lifesnaps_stressdetection_mlp_results.json).
The JSON contains split counts, split-key digests, configuration settings,
probability AUC, loss and timings; it contains no raw participant identifiers or
measurements. These values were rerun after the metric-validation helper was
added, and the final source-file hashes are included.

## What the ablations show, and what they do not show

The feature experiment asks whether the two input groups provide useful signal
under this small participant-held-out protocol. The combined and sensor-only
means are similar. Context-only varies substantially across the three splits;
its lower mean here does not establish that context is generally uninformative.
The small and imbalanced test sets make broad rankings unreliable.

The additional context-offset experiment asks how the **learned task with
older context** behaves. Context is selected from the same person's previous
available complete-day observations. It is not selected using a calendar-day
cutoff. When insufficient history exists, the first available record is used.
The entire context vector is shifted; static fields may therefore stay the same.
The earlier audit of the paper/reference code found no corresponding
observation-offset experiment, so this is an additional task-configuration
extension rather than a claim that modality ablation itself is new.

Every offset configuration is trained again, with its own training-only
normalization. The data split, sensor values, labels, architecture and epoch
budget remain fixed within a seed, but model weights do not. This design tests
sensitivity of the training task, **not** deployment-time robustness of a fixed
checkpoint. It is different from the independent reproduction's fixed-weight
offset experiment and from the proposal's stricter calendar-day experiment.

The observed means are not monotonically worse at larger offsets: offset 1 is
approximately unchanged and offset 7 is slightly higher on average. This pilot
therefore does not support a simple claim that older context always lowers MLP
AUC. Nor does it establish that delayed questionnaires are beneficial or safe,
that BERT pretraining is robust, or that a real weekly questionnaire schedule
would preserve performance. No significance test or paired uncertainty interval
has been performed, and no test score was used to choose a winning configuration.

## Calibration and software validation

The single real-data seed-0/both/one-epoch calibration took
1.482 seconds inside run_config, including
1.469 seconds for preparation and
0.012 seconds for training/validation/checkpoint selection.
These timers exclude interpreter and module-import startup. The final 15-run
matrix accumulated 6.519 seconds inside
run_config; it does not represent full process wall time or GPU-hours.

The synthetic all-configuration command succeeded. Four small example tests
check that validation/test values cannot alter normalization, empty splits are
rejected, single-class AUC is explicitly undefined, and nonfinite or mismatched
predictions are rejected. Dataset/Task and WearableStressModel tests are separate
from these MLP example checks. This script uses no author checkpoint and makes
no numerical-equivalence claim about the original CLSP architecture.
