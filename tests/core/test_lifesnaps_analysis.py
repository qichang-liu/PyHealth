"""Hand-calculated checks for the example's statistical analysis."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def helpers():
    path = Path(__file__).parents[2] / "examples/wearable/lifesnaps_stress_analysis.py"
    spec = importlib.util.spec_from_file_location("lifesnaps_stress_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auc_pair_decomposition_and_ties():
    module = helpers()
    result = module.auc_decomposition(
        [0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2], ["a", "a", "b", "b"]
    )
    assert result["overall_auc"] == 0.75
    assert result["within_auc"] == 0.5 and result["within_pairs"] == 2
    assert result["between_auc"] == 1 and result["between_pairs"] == 2
    tied = module.auc_decomposition([0, 1], [0.2, 0.2], ["a", "b"])
    assert tied["overall_auc"] == 0.5
    assert tied["within_auc"] is None and tied["within_pairs"] == 0


def test_paired_bootstrap_retains_identical_conditions():
    scores = np.array([[0.1, 0.9, 0.8, 0.2]] * 2)
    result = helpers().paired_cluster_bootstrap(
        [0, 1, 0, 1], scores, ["a", "a", "b", "b"], [[1, -1]], n_resamples=200, seed=7
    )
    assert result["estimates"] == [0]
    assert result["lower_95"] == [0] == result["upper_95"]
    assert result["valid_resamples"] == 200
    # Each cluster has both classes, so no bootstrap draw is single-class.
    assert result["excluded_single_class_resamples"] == 0


def test_single_class_bootstrap_draws_are_reported():
    result = helpers().paired_cluster_bootstrap(
        [0, 1], [[0.1, 0.9]], ["a", "b"], [[1]], n_resamples=200, seed=7
    )
    assert 0 < result["excluded_single_class_resamples"] < 200
    assert result["lower_95"] == [1] == result["upper_95"]
    with pytest.raises(ValueError, match="both classes"):
        helpers().paired_cluster_bootstrap([0, 0], [[0.1, 0.9]], ["a", "b"], [[1]])


def test_invalid_arrays_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        helpers().auc_decomposition([0, 1], [0.1, np.inf], ["a", "b"])
    with pytest.raises(ValueError, match="Contrast"):
        helpers().paired_cluster_bootstrap([0, 1], [[0.1, 0.9]], ["a", "b"], [[1, -1]])


def test_cluster_bootstrap_matches_explicit_resampled_people():
    from sklearn.metrics import roc_auc_score

    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.6])
    ids = np.array(["a", "a", "b", "b", "c", "c"])
    result = helpers().paired_cluster_bootstrap(
        labels, scores, ids, [[1]], n_resamples=30, seed=12
    )
    weights = np.random.default_rng(12).multinomial(3, [1 / 3] * 3, size=30)
    explicit = []
    for draw in weights:
        rows = np.repeat(np.arange(6), np.repeat(draw, 2))
        explicit.append(roc_auc_score(labels[rows], scores[rows]))
    np.testing.assert_allclose(result["estimates"], [roc_auc_score(labels, scores)])
    np.testing.assert_allclose(
        [result["lower_95"][0], result["upper_95"][0]],
        np.percentile(explicit, [2.5, 97.5]),
    )
