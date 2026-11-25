"""Analyze controlled LifeSnaps runs for the proposal's planned experiments.

Authors: Xiangwen Wang (xw120@illinois.edu), Qichang Liu (qichang4@illinois.edu).
Paper: https://proceedings.mlr.press/v287/yang25a.html

Usage from the repository root::

    python examples/wearable/lifesnaps_stressdetection_analysis.py \
        --runs-root /path/to/runs --output /tmp/lifesnaps-analysis

Expect reference_seed0, stop_gradient_seed0, one_hot_seed0, random_seed0,
statistics_seed0 and corresponding seed1/seed2 directories. Each contains
metrics.json and preprocessing.json from the wearable stress example. The
random/statistics runs also contain evaluation labels and cluster IDs in their
local metrics JSON; those identifiers are never written into aggregate CSVs.
"""

import argparse
import csv
import json
from pathlib import Path

import lifesnaps_stress_analysis as analysis
import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    """Check matched inputs and write aggregate, identifier-free analyses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261125)
    args = parser.parse_args()
    base = args.runs_root
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        "supervised",
        "reference",
        "stop_gradient",
        "one_hot",
        "random",
        "statistics",
    ]
    decomposition = []
    uncertainty = []
    performance = []
    for seed in range(3):
        files = {key: base / f"{key}_seed{seed}/metrics.json" for key in methods[1:]}
        runs = {key: json.loads(path.read_text()) for key, path in files.items()}
        metadata = runs["random"]["evaluation"]
        labels = np.asarray(metadata["labels"])
        ids = np.asarray(metadata["patient_ids"])
        reference = runs["reference"]
        ref_probs = reference["models"]["supervised"]["test"]["probabilities"]
        assert metadata == runs["statistics"]["evaluation"]
        for name, d in runs.items():
            assert d["split"] == reference["split"]
            assert d["pretraining_days"] == 256
            if "evaluation" in d:
                assert d["evaluation"] == metadata
            np.testing.assert_allclose(
                d["models"]["supervised"]["test"]["probabilities"],
                ref_probs,
                atol=0,
                rtol=0,
            )
            assert (base / f"{name}_seed{seed}/preprocessing.json").read_bytes() == (
                base / f"reference_seed{seed}/preprocessing.json"
            ).read_bytes()
        scores = {}
        for method in methods:
            d = reference if method == "supervised" else runs[method]
            model = d["models"]["supervised" if method == "supervised" else "clsp"]
            for item in model["calendar_context_ablation"]:
                delay = item["calendar_days"]
                probs = np.asarray(item["probabilities"])
                np.testing.assert_allclose(
                    roc_auc_score(labels, probs), item["roc_auc"], atol=1e-12
                )
                scores[method, delay] = probs
                result = analysis.auc_decomposition(labels, probs, ids)
                if result["within_pairs"]:
                    reconstructed = (
                        result["within_auc"] * result["within_pairs"]
                        + result["between_auc"] * result["between_pairs"]
                    ) / result["overall_pairs"]
                    np.testing.assert_allclose(
                        reconstructed, item["roc_auc"], atol=1e-12
                    )
                decomposition.append(
                    {
                        "seed": seed,
                        "method": method,
                        "calendar_days": delay,
                        "test_samples": len(labels),
                        "test_participants": len(np.unique(ids)),
                        **result,
                    }
                )
                performance.append(
                    {
                        "seed": seed,
                        "method": method,
                        "calendar_days": delay,
                        "roc_auc": item["roc_auc"],
                        "hard_label_auc": item["hard_label_auc"],
                        "fallback_count": item["fallback_count"],
                    }
                )
        for delay in [1, 3, 7]:
            matrix = np.stack(
                [
                    scores["supervised", 0],
                    scores["supervised", delay],
                    scores["reference", 0],
                    scores["reference", delay],
                ]
            )
            names = [
                "supervised_drop",
                "reference_drop",
                "reference_minus_supervised_drop",
            ]
            boot = analysis.paired_cluster_bootstrap(
                labels,
                matrix,
                ids,
                [[1, -1, 0, 0], [0, 0, 1, -1], [-1, 1, 1, -1]],
                n_resamples=args.resamples,
                seed=args.bootstrap_seed + seed,
            )
            for i, name in enumerate(names):
                uncertainty.append(
                    {
                        "seed": seed,
                        "comparison": name,
                        "calendar_days": delay,
                        "estimate": boot["estimates"][i],
                        "lower_95": boot["lower_95"][i],
                        "upper_95": boot["upper_95"][i],
                        **{
                            key: boot[key]
                            for key in [
                                "clusters",
                                "requested_resamples",
                                "valid_resamples",
                                "excluded_single_class_resamples",
                            ]
                        },
                    }
                )
        for method in [
            "supervised",
            "stop_gradient",
            "one_hot",
            "random",
            "statistics",
        ]:
            boot = analysis.paired_cluster_bootstrap(
                labels,
                np.stack([scores[method, 0], scores["reference", 0]]),
                ids,
                [[1, -1]],
                n_resamples=args.resamples,
                seed=args.bootstrap_seed + seed,
            )
            uncertainty.append(
                {
                    "seed": seed,
                    "comparison": method + "_minus_reference_auc",
                    "calendar_days": 0,
                    "estimate": boot["estimates"][0],
                    "lower_95": boot["lower_95"][0],
                    "upper_95": boot["upper_95"][0],
                    **{
                        key: boot[key]
                        for key in [
                            "clusters",
                            "requested_resamples",
                            "valid_resamples",
                            "excluded_single_class_resamples",
                        ]
                    },
                }
            )
    for name, rows in [
        ("proposal_performance", performance),
        ("proposal_auc_decomposition", decomposition),
        ("proposal_bootstrap", uncertainty),
    ]:
        output = output_dir / f"lifesnaps_{name}.csv"
        with output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        method: {
            str(delay): {
                "mean": float(
                    np.mean(
                        v := [
                            r["roc_auc"] * 100
                            for r in performance
                            if r["method"] == method and r["calendar_days"] == delay
                        ]
                    )
                ),
                "sd": float(np.std(v)),
            }
            for delay in [0, 1, 3, 7]
        }
        for method in methods
    }
    (output_dir / "proposal_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(
        "All matched controls, AUC identities, paired participant bootstrap draws checked."
    )
    print("CSV rows:", len(performance), len(decomposition), len(uncertainty))


if __name__ == "__main__":
    main()
