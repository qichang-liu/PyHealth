# Controlled contrastive-target ablation

Measured 2026-11-25. Authors: Xiangwen Wang (xw120@illinois.edu), Qichang Liu (qichang4@illinois.edu).

[Original paper](https://proceedings.mlr.press/v287/yang25a.html) · [Example](lifesnaps_stressdetection_wearable_stress.py) · [Aggregate runs](lifesnaps_objective_ablation.csv) · [All proposal analyses](lifesnaps_proposal_findings.md)

The original local implementation detached its soft targets. That was an implementation deviation, not a previously planned ablation. We corrected the default to differentiable soft targets, retained an explicit legacy mode, and then ran this controlled comparison. Historical results remain labeled as stop-gradient results; they were not overwritten or relabeled.

## What changes

For projected multimodal and text batches U and V, the retained reference forward formula uses logits U Vᵀ / temperature and soft targets softmax((U Uᵀ + V Vᵀ) × temperature / 2). Cross-entropies in both directions are averaged.

| Mode | Target | Gradient through target | Purpose |
|---|---|---|---|
| reference (current default) | Similarity-derived soft distribution | Yes | Match the reference code's target-gradient behavior |
| stop_gradient | Same soft distribution | No | Quantify the earlier implementation deviation |
| one_hot | Only the matching daily example is positive | Not applicable | Compare the one-hot target described in the paper within this adapted pipeline |

Reference and stop-gradient have identical loss values for identical input tensors, but generally different gradients. Tests compare both losses and gradients to independent explicit formulas; all three modes survive checkpoint restoration. Old checkpoints without a target-mode field load as stop_gradient. Fixing this objective does not remove remaining architecture, preprocessing, prompt, split or training-budget differences from the paper. The one_hot experiment changes the target, not every aspect of the published method.

## Matched protocol and measured results

Nine actual runs: three target modes × seeds 0/1/2. Within each seed, participant splits, preprocessing, selected 256 training days, BERT embeddings, initialization and budget are fixed. The supervised baseline's predictions match exactly across modes. The corresponding preprocessing files and BERT cache identities also match. Seeds vary both split and initialization.

Participant-level train/validation/test = 60/20/20; frozen bert-base-uncased; CPU; supervised 30 epochs; pretrain 30, probe 10, fine-tune 20; Adam lr=0.001, batch=512, dropout=0.1. Classification checkpoints use validation probability AUC. No test-based configuration choice. See the historical pilot for the same split counts and data preparation; its target mode is explicitly different.

Probability ROC-AUC %, mean ± population SD across three seeds (not a confidence interval):

| Method | k=0 | k=1 | k=3 | k=7 |
|---|---|---|---|---|
| supervised | 52.83 ± 3.51 | 56.31 ± 2.67 | 57.47 ± 2.48 | 50.00 ± 6.04 |
| reference | 50.03 ± 7.65 | 58.95 ± 8.08 | 51.99 ± 9.85 | 51.44 ± 10.18 |
| stop_gradient | 50.08 ± 12.98 | 56.20 ± 8.31 | 54.52 ± 13.93 | 49.27 ± 9.95 |
| one_hot | 47.55 ± 9.17 | 55.81 ± 8.24 | 52.60 ± 14.32 | 47.86 ± 9.33 |

Neither target comparison establishes a consistent performance advantage. The tiny 0.05 percentage-point difference between reference and stop-gradient means is not evidence of equivalent training. Individual seeds differ substantially. We retain reference as the default for its stated behavioral correspondence, not because it maximizes these test results.

Calendar-day columns evaluate the same trained weights and test rows, changing only time-varying context. They are not additional retraining runs. The CSV has 48 aggregate conditions: four methods × four delays × three seeds. Supervised is included once per seed, not counted as three independent baselines.

## Reproduce

From the PyHealth repository root, use the public LifeSnaps release and locally cached bert-base-uncased. Substitute your data path. Run for each seed 0, 1, 2 and each mode reference, stop_gradient, one_hot; choose fresh output directories.

```sh
python examples/wearable/lifesnaps_stressdetection_wearable_stress.py \
  --root /path/to/rais_anonymized --output /tmp/runs/reference_seed0 \
  --seed 0 --device cpu --max-pretrain-days 256 --text-batch-size 16 \
  --text-encoder pretrained --contrastive-target reference \
  --pretrain-epochs 30 --probe-epochs 10 --finetune-epochs 20 \
  --supervised-epochs 30 --batch-size 512 --lr 0.001 --dropout 0.1
```

The pretrained BERT revision used here is 86b5e0934494bd15c9632b12f734a8a67f723594; no selected prompt was truncated. Local checkpoints and per-day predictions are not PR artifacts. Aggregate CSVs and these settings are included for review.
