# LifeSnaps complete PyHealth CLSP pipeline: measured pilot results

Run date: 2026-11-25. Authors: Xiangwen Wang (xw120@illinois.edu), Qichang Liu (qichang4@illinois.edu).

[Original paper](https://proceedings.mlr.press/v287/yang25a.html) · [Example](lifesnaps_stressdetection_wearable_stress.py) · [Helpers](lifesnaps_clsp_utils.py) · [Per-run metrics CSV](lifesnaps_clsp_results.csv)

These are actual measurements from the contributed PyHealth Dataset, Task, and WearableStressModel. They are distinct from the earlier independent reproduction. This is a limited-compute pilot, not a complete or numerically identical reproduction of the paper.

Historical snapshot: these runs used `contrastive_target=stop_gradient`. The
current default is `reference` (differentiable soft targets); pass the explicit
legacy flag below to reproduce this table. The target-mode ablation is recorded
separately in `lifesnaps_objective_ablation.md`; these numbers are not relabeled.

## Protocol and controlled comparison

- Data: LifeSnaps, 71 participants, 5,850 complete participant-days, 228 labeled days. The pretraining Task also exposes complete unlabeled days without a stress field.
- Predeclared seeds: 0, 1, 2. Each seed creates one participant-level 60/20/20 train/validation/test split. All methods within a seed share the same participants and labeled evaluation rows. Seeds change both the split and initialization; three-seed variability is not a confidence interval.
- Only training participants fit the coordinate-wise quantile preprocessing. It includes sensor/hour coordinates and all context coordinates, including deterministic task category codes. No validation/test participant enters normalization fitting or contrastive pretraining.
- For each seed, sample 256 complete training days without replacement using that seed. Encode their allowed raw context text and 11 raw statistics per sensor channel with frozen bert-base-uncased, revision 86b5e0934494bd15c9632b12f734a8a67f723594. Embeddings use attention-masked mean pooling including special tokens. Context appears first; max_length=512. No selected prompt was truncated in these runs.
- Supervised baseline: 30 epochs. CLSP adaptation: 30 contrastive epochs, 10 frozen-probe epochs, then 20 fine-tuning epochs. Adam learning rate 0.001, batch size 512, dropout 0.1. Classification stages restore their best validation probability ROC-AUC; no test-based configuration selection.
- The contrastive soft-target forward formula is informed by reference code, but target probabilities are detached here. The authors' reference has differentiable targets and the paper states a one-hot objective. This stop-gradient choice is an explicit adaptation, not a claim of gradient-level equivalence.
- Each trained model is saved and reloaded. Held-out predictions agree within absolute tolerance 1e-6 after checkpoint restoration. Preprocessing parameters are separately saved and roundtrip-checked.
- Calendar-day ablation: for fixed test rows and weights, replace dynamic context with the latest same-person record dated no later than t-k, k in {0,1,3,7}. Static demographics/personality, sensor inputs, labels and model weights stay fixed. If no eligible history exists, use training-partition coordinate medians for dynamic fields. These runs had zero fallbacks for all reported conditions; the fallback path is covered by a synthetic unit test.

## Dataset split sizes

| Seed | Train labeled / positive | Validation labeled / positive | Test labeled / positive | Training complete days | CLSP pretraining subset |
|---|---|---|---|---|---|
| 0 | 133 / 46 | 58 / 21 | 37 / 13 | 3770 | 256 |
| 1 | 139 / 45 | 43 / 24 | 46 / 11 | 3411 | 256 |
| 2 | 129 / 51 | 60 / 15 | 39 / 14 | 3502 | 256 |

## Probability ROC-AUC results

Values below are percentages, mean ± population SD over the three predeclared seeds. Individual runs and hard-prediction AUC are in the CSV.

| Model | k=0 calendar days | k=1 | k=3 | k=7 |
|---|---|---|---|---|
| supervised | 52.83 ± 3.51 | 56.31 ± 2.67 | 57.47 ± 2.48 | 50.00 ± 6.04 |
| clsp | 50.08 ± 12.98 | 56.20 ± 8.31 | 54.52 ± 13.93 | 49.27 ± 9.95 |

## Per-seed comparisons and interpretation

| Seed | Supervised k=0 | CLSP k=0 | Supervised k=7 minus k=0 (pp) | CLSP k=7 minus k=0 (pp) |
|---|---|---|---|---|
| 0 | 52.24 | 67.63 | -10.26 | -7.05 |
| 1 | 57.40 | 36.62 | -5.97 | -0.26 |
| 2 | 48.86 | 46.00 | +7.71 | +4.86 |

The effect of pretraining and context delay varies across seeds. The comparison therefore does not establish a consistent accuracy or robustness advantage. The delay experiment is an implemented calendar-day intervention, whereas the older independent experiments and the Task's historical offset parameter count observations. Do not merge these two experiments into one table.

No paired participant-cluster confidence interval is claimed. Small test sets, participant heterogeneity, the limited training budget, category encoding, prompt construction and the stop-gradient objective all limit attribution. Context-delay changes need not be monotonic. Higher performance in an individual delayed condition does not establish a deployment recommendation.

## Runtime calibration and execution

All runs used CPU because MPS was unavailable in the execution environment. A complete 64-day real-BERT, one-epoch-per-stage calibration ran first: BERT encoding 6.05 seconds; whole measured pipeline 10.12 seconds. The 256-day three-seed budget was selected after that calibration, before inspecting its test scores.

| Seed | BERT seconds | Supervised training seconds | CLSP training seconds | Whole measured pipeline seconds |
|---|---|---|---|---|
| 0 | 20.45 | 4.55 | 10.20 | 38.78 |
| 1 | 19.63 | 3.92 | 9.55 | 36.68 |
| 2 | 20.04 | 3.89 | 9.42 | 36.83 |

Timings are measured inside the example and exclude interpreter/import startup. The offline synthetic full-pipeline smoke also passed using random text embeddings; its AUC is not included as scientific evidence.

## Reproduce

From the PyHealth repository root, prepare the public LifeSnaps release and make bert-base-uncased available in the Hugging Face cache (or explicitly add --allow-download). Then run the following command for each seed 0, 1, 2, keeping every other argument fixed. Substitute your local data path and output directory.

```sh
python examples/wearable/lifesnaps_stressdetection_wearable_stress.py \
  --root /path/to/rais_anonymized --output /tmp/lifesnaps-clsp-seed0 \
  --seed 0 --device cpu --max-pretrain-days 256 --text-batch-size 16 \
  --pretrain-epochs 30 --probe-epochs 10 --finetune-epochs 20 \
  --supervised-epochs 30 --batch-size 512 --lr 0.001 --dropout 0.1 \
  --contrastive-target stop_gradient
```

Each output directory contains metrics.json (per-epoch validation/loss, held-out predictions and delay results), split.json, preprocessing.json, supervised.pt and clsp.pt. Raw participant-level artifacts and model weights are local outputs, not committed in this contribution. The CSV contains only aggregate run metrics, sufficient to recompute the tables.

Comparison with the paper must state protocol differences: this participant-held-out, reduced-pretraining pilot differs from the paper's random-day Table 1 setting and the previous independent reproduction. The standalone MLP example also has a distinct preprocessing/split construction; its scores should not be silently combined into this matched two-model comparison.
