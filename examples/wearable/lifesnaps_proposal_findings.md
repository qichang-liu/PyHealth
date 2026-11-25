# Proposal experiments: calendar context, uncertainty and text mechanisms

Measured 2026-11-25. Authors: Xiangwen Wang (xw120@illinois.edu), Qichang Liu (qichang4@illinois.edu).

[Original paper](https://proceedings.mlr.press/v287/yang25a.html) · [Training example](lifesnaps_stressdetection_wearable_stress.py) · [Analysis CLI](lifesnaps_stressdetection_analysis.py) · [Analysis helpers](lifesnaps_stress_analysis.py)

All planned LifeSnaps analyses described below were executed. Their results do not establish a consistent CLSP performance or stale-context robustness advantage. Completing an experiment is distinct from confirming its hypothesis. These are participant-held-out, limited-compute PyHealth results; independent random-day reproduction results must be reported separately.

## Mapping the proposal to completed evidence

| Planned item | Implementation and evidence | Result |
|---|---|---|
| Dataset + Task, documentation and synthetic tests | LifeSnapsDataset, StressDetectionLifeSnaps, API RST/indexes, test_lifesnaps.py | CSV → event → supervised/unlabeled samples verified |
| MLP feature/configuration ablation | lifesnaps_stressdetection_mlp.py and findings/results JSON | 15 measured runs: five configurations × three seeds |
| Optional model integration, now implemented | WearableStressModel and complete training example | Real paired text, contrastive pretraining, frozen probing, fine-tuning, save/reload |
| Calendar k=0/1/3/7 | calendar context helper; fixed model and test rows | All delays evaluated; zero real-data fallbacks; synthetic fallback behavior tested |
| Paired participant-cluster bootstrap | lifesnaps_stress_analysis.py and bootstrap CSV | 2,000 paired draws per seed; delay comparisons and intervals below |
| Within-/between-participant AUC | AUC decomposition CSV | Pair-weighted decomposition; counts reported |
| Frozen pretrained / random BERT / statistical projection | --text-encoder pretrained/random/statistics | All three controls × three seeds measured |

## Shared experimental protocol

The [objective ablation](lifesnaps_objective_ablation.md) specifies splits, hyperparameters and checkpoint selection. Reuse its three reference/pretrained runs; add six runs for random/statistics. All text controls use the reference target mode. The overall matrix contains 15 CLSP training runs (nine objective and six additional text controls), not 18. Each run also trains a matched supervised baseline; repeated identical supervised predictions are counted once. Calendar evaluation produces 72 aggregate conditions for six distinct methods × four delays × three seeds.

Only training participants fit normalization or enter pretraining. Each seed selects 256 complete training days. Fixed epochs: supervised30/pretrain30/probe10/fine-tune20, lr0.001, batch512, dropout0.1. The three test partitions contain 37/46/39 labeled days and 9/10/9 participants with labels. Three-seed variability mixes split and initialization variability; this small experiment has limited precision.

## Text mechanism controls

- **pretrained/reference**: frozen bert-base-uncased, attention-masked mean pooling, max_length512, context before sensor statistics, same tokenizer and prompts across comparable BERT runs. No selected prompt truncation.
- **random**: same BERT configuration and tokenizer, but every model weight is randomly initialized with fixed encoder seed20261125 and then frozen. This is a full randomly initialized BERT, not random output vectors. Its separate cache identity prevents reuse of pretrained embeddings.
- **statistics**: direct learned projection of 77 underlying numerical features (11 statistics × five sensor channels +22 context coordinates), standardized with only the selected training subset. Zero-padding to768 reuses the text projection interface; padded coordinates add no input information. No BERT is used. This branch includes the same sensor summary quantities and contextual variables, but category representation and normalization differ from textual encoding, so it is not a pure semantics-only intervention.

Probability ROC-AUC %, mean ± population SD over three seeds:

| Method | k=0 | k=1 | k=3 | k=7 |
|---|---|---|---|---|
| supervised | 52.83 ± 3.51 | 56.31 ± 2.67 | 57.47 ± 2.48 | 50.00 ± 6.04 |
| reference | 50.03 ± 7.65 | 58.95 ± 8.08 | 51.99 ± 9.85 | 51.44 ± 10.18 |
| random | 52.50 ± 19.80 | 57.97 ± 14.74 | 53.94 ± 25.01 | 51.56 ± 20.01 |
| statistics | 45.13 ± 8.82 | 44.71 ± 10.78 | 46.32 ± 12.14 | 42.96 ± 3.29 |

Pretrained BERT does not demonstrate a stable advantage over random frozen BERT in this small matrix. Random BERT has very high cross-seed variation. The statistics baseline performs worse on average, but these results do not prove that language semantics causes the difference, or that pretrained semantics is useless. Training budget, dimensionality and representation choices remain relevant limitations.

## Fixed-weight calendar intervention and paired intervals

At delay k, replace only dynamic context by the latest same-participant observation dated no later than t−k. Keep sensors, labels, static attributes, trained weights and evaluation rows fixed. The predefined no-history fallback uses training-partition coordinate medians; counts are zero for these real-data conditions and are included in the performance CSV. Historical observation-offset experiments answer a different question and are not relabeled as calendar days.

AUC drop = AUC(k=0) − AUC(k). Positive means deterioration; negative means improvement. The contrast of interest is CLSP drop minus supervised drop; a negative value favors CLSP robustness. For each training seed, sample participants with replacement and retain every labeled day belonging to each sampled participant. Apply exactly the same sampled participant multiplicities to both models and both delays. Report percentile95% intervals from2,000 draws. Single-class draws cannot define AUC and are excluded: seed0 has2,000 valid, seed1 has1,986 valid/14 excluded, seed2 has2,000 valid. These intervals condition on already fitted models and the selected split, not on retraining. Overlapping splits are not pooled as independent observations.

Seven-calendar-day comparison:

| Seed | Supervised AUC drop (pp) | Reference AUC drop (pp) | Reference minus supervised drop, 95% CI (pp) |
|---|---|---|---|
| 0 | +10.26 | -1.28 | -11.54 [-30.94, +1.76] |
| 1 | +5.97 | +3.90 | -2.08 [-41.29, +42.50] |
| 2 | -7.71 | -6.86 | +0.86 [-9.85, +13.17] |

Every reference-minus-supervised drop interval at k=1,3,7 includes zero. We therefore do not establish a consistent robustness advantage, a monotonic degradation pattern, or equivalence between methods. All 42 interval records, including five method-versus-reference comparisons per seed at k=0, are in [the bootstrap CSV](lifesnaps_proposal_bootstrap.csv).

## Within- and between-participant AUC

For each positive-negative pair, count whether the positive's score exceeds the negative's (ties get half credit). Separate pairs from the same participant and from different participants. These component AUCs are pair-weighted, not an unweighted mean of per-person AUCs. A component without eligible pairs is undefined, not zero. The pair-count-weighted sum reconstructs global AUC and is checked automatically.

| Seed | Model | Overall AUC % | Within AUC % / pairs | Between AUC % / pairs |
|---|---|---|---|---|
| 0 | supervised | 52.24 | 68.18 / 22 | 51.03 / 290 |
| 0 | reference | 59.94 | 54.55 / 22 | 60.34 / 290 |
| 1 | supervised | 57.40 | 50.00 / 22 | 57.85 / 363 |
| 1 | reference | 41.30 | 54.55 / 22 | 40.50 / 363 |
| 2 | supervised | 48.86 | 36.00 / 25 | 49.85 / 325 |
| 2 | reference | 48.86 | 40.00 / 25 | 49.54 / 325 |

About93–94% of eligible pairs are between people. Thus global AUC is dominated by cross-person discrimination and does not by itself establish reliable tracking of day-to-day changes within one person. Within-person estimates have only22–25 pairs per seed and should not be treated as precise. This is a diagnostic, not causal proof of dataset bias. The motivation is related to concerns about personal baselines in [DeMasi et al. (2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0184604); the specific pair decomposition here is our secondary analysis, not a method attributed to that paper.

## Reproduce the text controls and analysis

Run the objective commands for seeds0/1/2 into reference_seedN, stop_gradient_seedN, one_hot_seedN. For each seed, run the same command with --contrastive-target reference and --text-encoder random or statistics; name outputs random_seedN/statistics_seedN. Keep every other argument fixed. random_encoder_seed defaults to20261125. Then:

```sh
python examples/wearable/lifesnaps_stressdetection_analysis.py \
  --runs-root /tmp/runs --output /tmp/analysis \
  --resamples 2000 --bootstrap-seed 20261125
```

The CLI checks matching splits, pretraining budgets, preprocessing, evaluation metadata when available, supervised predictions, and stored/recomputed AUCs. It outputs three identifier-free CSVs and proposal_summary.json. We ran this contributed CLI separately and reproduced the committed aggregates within1e-12. Small hand-calculated tests check ties, absent within-person pairs, paired contrasts, single-class exclusions, and agreement with explicit participant resampling plus sklearn ROC-AUC. No real data or BERT download is needed for these unit tests.

- [Performance:72 rows](lifesnaps_proposal_performance.csv)
- [AUC decomposition:72 rows](lifesnaps_proposal_auc_decomposition.csv)
- [Paired bootstrap:42 rows](lifesnaps_proposal_bootstrap.csv)

Per-day predictions, participant IDs, caches, original data and checkpoints stay local and are excluded from this PR. Full Sphinx build and external PR/CI acceptance are separate from these measured scientific results.
