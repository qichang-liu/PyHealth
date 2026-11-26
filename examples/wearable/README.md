# LifeSnaps stress detection in PyHealth

This contribution implements a reusable **Dataset + Task + Model** pipeline for
daily stress detection from wearable sensors and contextual features. It includes
synthetic tests, runnable examples, measured ablations, and participant-level
uncertainty analyses.

**Paper:** Yang, Yu, and Sano (CHIL 2025),
[Contrastive Pretraining for Stress Detection with Multimodal Wearable Sensor Data and Surveys](https://proceedings.mlr.press/v287/yang25a.html).

**Contributors:** Xiangwen Wang (`xw120@illinois.edu`) and Qichang Liu
(`qichang4@illinois.edu`).

The implementation and experiments run locally. The results below do **not**
establish a consistent advantage of contrastive pretraining. This is a documented
PyHealth adaptation, not numerical equivalence to every experiment in the paper.

## What is implemented

| Component | Implementation | Responsibility |
|---|---|---|
| Dataset | [LifeSnapsDataset](../../pyhealth/datasets/lifesnaps.py) and [YAML](../../pyhealth/datasets/configs/lifesnaps.yaml) | Raw hourly/daily files to standard PyHealth events; complete-day filtering, deduplication, missing-label handling |
| Task | [StressDetectionLifeSnaps](../../pyhealth/tasks/stress_detection_lifesnaps.py) | Five sensor channels × 24 hours, 22 context coordinates, binary stress labels or unlabeled pretraining samples |
| Model | [WearableStressModel](../../pyhealth/models/wearable_stress.py) | Sensor Transformer, residual context encoder, fusion/projections, supervised/pretrain/probe/finetune stages and checkpoint API |
| MLP example | [lifesnaps_stressdetection_mlp.py](lifesnaps_stressdetection_mlp.py) | Feature and task-configuration comparisons using the existing PyHealth MLP |
| Full pipeline | [lifesnaps_stressdetection_wearable_stress.py](lifesnaps_stressdetection_wearable_stress.py) | Paired text, frozen BERT, contrastive training, probing, fine-tuning, evaluation and calendar-delay controls |
| Analysis | [lifesnaps_stressdetection_analysis.py](lifesnaps_stressdetection_analysis.py) | Matched-run checks, paired participant bootstrap and within-/between-participant AUC |

STAI and label-derived fields are excluded from numerical and textual inputs.
Text is constructed from allowed context and sensor statistics from the same
daily record; it is not a separate clinical report. Text/BERT is used during
pretraining. Downstream classification consumes sensors and context.

The Dataset also includes a shared-library change: `BaseDataset(num_workers=1)`
executes the existing Dask event-processing graph synchronously, without starting
a local server. The multiworker path is retained.

## Install and run the offline checks

Use Python **3.12 or 3.13**, as required by this checkout's `pyproject.toml`.
Run commands from the repository root. For this review branch:

```sh
git clone --branch lifesnaps-stress-detection git@github.com:qichang-liu/PyHealth.git
cd PyHealth
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[lint]' pytest pytest-subtests
```

Run the relevant synthetic tests:

```sh
python -m pytest \
  tests/core/test_lifesnaps.py \
  tests/core/test_wearable_stress.py \
  tests/core/test_lifesnaps_clsp_example.py \
  tests/core/test_lifesnaps_mlp_example.py \
  tests/core/test_lifesnaps_analysis.py \
  tests/core/test_base_dataset.py \
  tests/core/test_dataset_cache_key.py \
  -q --tb=short --durations=0
```

Latest recorded acceptance run: **47 passed, 3 subtests passed in 14.82 seconds**,
including imports/collection. Maximum test call was 0.26 seconds; maximum setup
was 0.68 seconds. One dependency-version notification was reported. Times depend
on the machine. No real data, BERT download, or pretrained checkpoint is required
for these tests.

The tests use four artificial participants and small tensors. They check exact
event/sample values, complete-day filtering, missing/duplicate handling, labels,
feature configurations, gradient flow, frozen probing, target-loss formulas,
checkpoint restoration, train-only preprocessing and hand-calculated AUC cases.
These checks establish software behavior, not predictive performance.

Run the two complete offline examples:

```sh
python examples/wearable/lifesnaps_stressdetection_mlp.py --synthetic
python examples/wearable/lifesnaps_stressdetection_wearable_stress.py \
  --synthetic --device cpu --output /tmp/lifesnaps-smoke
```

Synthetic text vectors are deterministic random inputs for software validation.
Their AUC is not scientific evidence about BERT or stress prediction.

## Prepare real data and run a baseline

Download `rais_anonymized.zip` from the
[LifeSnaps release](https://zenodo.org/records/7229547), extract it, and pass the
`rais_anonymized` directory to `--root`. Required source files include:

```text
rais_anonymized/
  csv_rais_anonymized/hourly_fitbit_sema_df_unprocessed.csv
  scored_surveys/personality.csv
  scored_surveys/stai.csv
```

Our prepared data contain 71 participants, 5,850 complete participant-days and
228 labeled days. The task maps `Above average` STAI category to 1 and
`Average`/`Below average` to 0; missing labels remain unlabeled. The loader writes
prepared tables, so use a writable data directory.

The real text example expects `bert-base-uncased` in the local Hugging Face cache.
Add `--allow-download` to explicitly permit fetching missing model/tokenizer files.
The measured pretrained runs used revision
`86b5e0934494bd15c9632b12f734a8a67f723594`.

```sh
python examples/wearable/lifesnaps_stressdetection_wearable_stress.py \
  --root /path/to/rais_anonymized --output /tmp/lifesnaps-reference-seed0 \
  --seed 0 --device cpu --max-pretrain-days 256 --text-batch-size 16 \
  --text-encoder pretrained --contrastive-target reference \
  --pretrain-epochs 30 --probe-epochs 10 --finetune-epochs 20 \
  --supervised-epochs 30 --batch-size 512 --lr 0.001 --dropout 0.1
```

Each run trains a supervised comparator and a contrastively pretrained model.
Participants are split 60/20/20; only training participants fit preprocessing and
enter pretraining. Classification checkpoints use validation probability AUC.
Saved checkpoints are reloaded and predictions are checked within tolerance.

Local outputs include `metrics.json`, `split.json`, `preprocessing.json`,
`supervised.pt` and `clsp.pt`. Metrics include per-day predictions and participant
identifiers for local paired analysis. Commit only the identifier-free aggregates,
not those raw run outputs, data, BERT caches or checkpoints.

## What the ablations test

An ablation changes a chosen input or training design while controlling the
other experimental settings, then measures the effect on held-out performance.

| Comparison | What changes | What stays controlled |
|---|---|---|
| MLP modalities | Both inputs / sensors only / context only | Same within-seed split, architecture and budget; each variant is retrained |
| MLP observation offsets | Context from earlier available observations | Same split/sensors/labels and budget; each configuration is retrained |
| CLSP target modes | Differentiable soft target / detached soft target / one-hot | Same within-seed split, pretraining days, text embeddings, initialization and budget |
| Text mechanisms | Pretrained frozen BERT / random frozen BERT / direct statistics projection | Same split, paired data, reference target and budget |
| Calendar delay | Dynamic context from the latest same-person date no later than t−k | Fixed trained weights, sensors, labels, static context and test rows |

**Context is still an input field in the sensors-only MLP.** Its 22 coordinates
are set to zero for every sample, preserving the interface and parameter count
while removing sample-specific context information. This is not a missing-field
error, nor a claim that the full CLSP baseline omits context.

The default `reference` target retains gradients through soft targets. The earlier
detached-target implementation is explicitly available as `stop_gradient`;
`one_hot` uses paired-index targets. Old checkpoints without a target-mode field
retain their original detached behavior. The target correction is documented in
[the objective findings](lifesnaps_objective_ablation.md).

`random` uses the full BERT architecture and tokenizer with randomly initialized,
frozen weights; it is not a random embedding lookup. `statistics` directly
projects 55 sensor statistics plus 22 context coordinates, using training-subset
standardization. Its representation/scaling differences limit a purely semantic
interpretation of that comparison.

## Measured results

All values are **probability ROC-AUC percentages**, mean ± population SD over
seeds 0/1/2. SD is not a confidence interval; AUC is not classification accuracy.
These small experiments do not establish a stable pretraining or robustness gain.

| CLSP-example configuration | Same-day context (k=0) | 7-calendar-day delay |
|---|---|---|
| Supervised | 52.83 ± 3.51 | 50.00 ± 6.04 |
| Reference target + pretrained BERT | 50.03 ± 7.65 | 51.44 ± 10.18 |
| Stop-gradient target | 50.08 ± 12.98 | 49.27 ± 9.95 |
| One-hot target | 47.55 ± 9.17 | 47.86 ± 9.33 |
| Reference + random frozen BERT | 52.50 ± 19.80 | 51.56 ± 20.01 |
| Reference + statistical projection | 45.13 ± 8.82 | 42.96 ± 3.29 |

The separate MLP protocol gives **69.50 ± 3.57** with both modalities,
**68.19 ± 1.70** with sensors only and **55.51 ± 18.08** with context only.
Its observation-offset configurations give **69.33 ± 0.89** (offset1) and
**70.79 ± 2.29** (offset7). MLP preprocessing, split construction and training
settings differ from the CLSP example: these two tables are not a matched
MLP-versus-CLSP comparison.

For calendar delays k=1/3/7, all paired participant-bootstrap intervals for
`CLSP AUC drop − supervised AUC drop` include zero. This does not establish
equivalence or superior CLSP robustness. Intervals use 2,000 participant-cluster
draws per seed, condition on fixed models/splits, and exclude/report single-class
draws. Within-/between-person decomposition shows that about 93–94% of eligible
positive-negative pairs are between people; global AUC alone does not demonstrate
reliable tracking of personal day-to-day stress.

| Evidence | Contents |
|---|---|
| [MLP findings](lifesnaps_stressdetection_findings.md) and [results JSON](lifesnaps_stressdetection_mlp_results.json) | Five configurations × three seeds, settings and provenance hashes |
| [Target findings](lifesnaps_objective_ablation.md) and [CSV](lifesnaps_objective_ablation.csv) | Nine target-mode training runs; supervised comparator and calendar conditions |
| [Proposal findings](lifesnaps_proposal_findings.md) | Text mechanisms, calendar controls, uncertainty and AUC decomposition |
| [Performance CSV](lifesnaps_proposal_performance.csv) | 72 aggregate evaluation conditions, including reused baselines |
| [Bootstrap CSV](lifesnaps_proposal_bootstrap.csv) | 42 paired contrasts with 95% intervals and valid/excluded draw counts |
| [AUC decomposition CSV](lifesnaps_proposal_auc_decomposition.csv) | 72 within-/between-person decompositions with pair counts |
| [Historical pilot](lifesnaps_clsp_results.md) | Earlier stop-gradient results, kept separately and labeled |

## Reproduce all controls and their analysis

For MLP feature/task controls:

```sh
python examples/wearable/lifesnaps_stressdetection_mlp.py \
  --root /path/to/rais_anonymized --epochs 20 --seeds 0 1 2 \
  --output /tmp/lifesnaps-mlp-results.json
```

For the CLSP matrix, use fresh output locations and the same data/cache throughout.
This shell example creates nine target-mode runs and six additional text-control
runs; it reuses the three reference/pretrained runs for the text comparison.

```sh
set -e
LIFESNAPS_ROOT=/path/to/rais_anonymized
LIFESNAPS_RUNS=/tmp/lifesnaps-controls
run_control() {
  python examples/wearable/lifesnaps_stressdetection_wearable_stress.py \
    --root "$LIFESNAPS_ROOT" --output "$LIFESNAPS_RUNS/${1}_seed${2}" \
    --seed "$2" --device cpu --max-pretrain-days 256 --text-batch-size 16 \
    --text-cache "$LIFESNAPS_RUNS/text_seed${2}" \
    --contrastive-target "$3" --text-encoder "$4" \
    --random-encoder-seed 20261125 \
    --pretrain-epochs 30 --probe-epochs 10 --finetune-epochs 20 \
    --supervised-epochs 30 --batch-size 512 --lr 0.001 --dropout 0.1
}
for seed in 0 1 2; do
  for target in reference stop_gradient one_hot; do
    run_control "$target" "$seed" "$target" pretrained
  done
  for encoder in random statistics; do
    run_control "$encoder" "$seed" reference "$encoder"
  done
done
python examples/wearable/lifesnaps_stressdetection_analysis.py \
  --runs-root "$LIFESNAPS_RUNS" --output /tmp/lifesnaps-analysis \
  --resamples 2000 --bootstrap-seed 20261125
```

The analysis checks matching splits, preprocessing and supervised predictions,
and reproduces aggregate CSVs from local run outputs. The contributed analysis
CLI was separately executed and reproduced the saved aggregates within 1e-12.
Four calendar conditions are evaluations of each trained model, not four new
training runs.

## Scope and review notes

- The participant-held-out, limited-pretraining PyHealth protocol differs from
  the paper's random-day setting. Remaining differences include category coding,
  normalization, text construction and the binary classification head. Matching
  the target gradient alone does not establish full methodological equivalence.
- A separate research reproduction also evaluated PMData. Its code and results
  are not bundled into this LifeSnaps contribution or relabeled as PyHealth runs.
- Review the three component APIs, their tests and example helpers; compare
  findings with the committed aggregates. Raw participant data and checkpoints
  are unnecessary for the synthetic acceptance tests.
- A full Sphinx build and multiworker execution were not validated in this work.
  Recorded local checks do not substitute for upstream review or CI.
- The authors' repository informed behavioral comparisons. No author package is
  imported by this contribution; remaining differences and historical corrections
  are disclosed in the linked findings.
