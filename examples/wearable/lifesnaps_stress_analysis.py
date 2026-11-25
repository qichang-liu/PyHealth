"""Paired participant-cluster uncertainty and within/between AUC analysis.

Authors: Xiangwen Wang (xw120@illinois.edu), Qichang Liu (qichang4@illinois.edu).
Paper: https://proceedings.mlr.press/v287/yang25a.html

These helpers analyze fixed held-out predictions. They do not fit models or
select configurations. Intervals condition on the fitted models and split;
they do not capture retraining or model-selection uncertainty.
"""

import numpy as np
from numpy.typing import ArrayLike


def _inputs(
    labels: ArrayLike, scores: ArrayLike, participants: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    participants = np.asarray(participants)
    if labels.ndim != 1 or participants.shape != labels.shape:
        raise ValueError("Labels and participant IDs must be aligned vectors")
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("Scores must have shape (conditions, samples)")
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Nonempty binary labels required")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    return labels, scores, participants


def auc_decomposition(
    labels: ArrayLike, scores: ArrayLike, participants: ArrayLike
) -> dict[str, float | int | None]:
    """Decompose global AUC by positive-negative pairs, giving ties half credit.

    Args:
        labels: Binary labels for held-out days.
        scores: One probability per day.
        participants: Participant identifier per day.

    Returns:
        Overall, within-person, and between-person AUC with their pair counts.
        A component with no eligible pairs is None, rather than zero. Components
        are pair-weighted, not an unweighted average of per-person AUCs.
    """
    labels, scores, participants = _inputs(labels, scores, participants)
    if len(scores) != 1:
        raise ValueError("Pass one prediction vector for decomposition")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    difference = scores[0, positive, None] - scores[0, negative][None, :]
    concordance = (difference > 0).astype(float) + 0.5 * (difference == 0)
    within = participants[positive, None] == participants[negative][None, :]
    result = {}
    for name, mask in [
        ("overall", np.ones_like(within)),
        ("within", within),
        ("between", ~within),
    ]:
        count = int(mask.sum())
        result[name + "_pairs"] = count
        result[name + "_auc"] = float(concordance[mask].mean()) if count else None
    return result


def paired_cluster_bootstrap(
    labels: ArrayLike,
    scores: ArrayLike,
    participants: ArrayLike,
    contrasts: ArrayLike,
    n_resamples: int = 2000,
    seed: int = 20261125,
) -> dict[str, int | list[float | None]]:
    """Percentile intervals for paired linear contrasts of probability AUCs.

    Args:
        labels: One binary label per held-out sample.
        scores: Shape (conditions, samples), all on the identical sample rows.
        participants: Cluster ID for each sample.
        contrasts: Shape (contrasts, conditions). For a drop use [1,-1];
            for difference of CLSP and supervised drops use [-1,1,1,-1]
            with rows [sup0,supk,clsp0,clspk].
        n_resamples: Number of participant bootstrap replicates.
        seed: Fixed resampling seed, independent of training seeds.

    Returns:
        Estimates and 95% percentile intervals in AUC units, replicate counts,
        and number of participant clusters. All conditions share each draw.
        Single-class draws are excluded and counted explicitly.
    """
    labels, scores, participants = _inputs(labels, scores, participants)
    contrasts = np.asarray(contrasts, dtype=float)
    if contrasts.ndim != 2 or contrasts.shape[1] != len(scores):
        raise ValueError("Contrast columns must match score conditions")
    if n_resamples < 1 or not np.isfinite(contrasts).all():
        raise ValueError("Positive resamples and finite contrasts required")
    clusters, inverse = np.unique(participants, return_inverse=True)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("The evaluation set must contain both classes")
    delta = scores[:, positive, None] - scores[:, negative][:, None, :]
    wins = (delta > 0).astype(float) + 0.5 * (delta == 0)
    point_aucs = wins.mean(axis=(1, 2))
    rng = np.random.default_rng(seed)
    # Multinomial multiplicities exactly reproduce cluster resampling with
    # replacement, retaining every day from every drawn participant.
    multiplicity = rng.multinomial(
        len(clusters), np.full(len(clusters), 1 / len(clusters)), size=n_resamples
    )
    positive_weights = multiplicity[:, inverse[positive]]
    negative_weights = multiplicity[:, inverse[negative]]
    denominator = positive_weights.sum(1) * negative_weights.sum(1)
    valid = denominator > 0
    numerator = np.einsum(
        "bp,cpn,bn->bc", positive_weights, wins, negative_weights, optimize=False
    )
    aucs = numerator[valid] / denominator[valid, None]
    # Small contrasts need only elementwise reductions. This also avoids
    # platform BLAS warnings observed for these narrow matrix products.
    samples = np.sum(aucs[:, None, :] * contrasts[None, :, :], axis=-1)
    estimates = np.sum(contrasts * point_aucs[None, :], axis=-1)
    lower, upper = (
        np.percentile(samples, [2.5, 97.5], axis=0)
        if len(samples)
        else (np.full(len(contrasts), np.nan),) * 2
    )
    return {
        "estimates": estimates.tolist(),
        "lower_95": [float(x) if np.isfinite(x) else None for x in lower],
        "upper_95": [float(x) if np.isfinite(x) else None for x in upper],
        "clusters": len(clusters),
        "requested_resamples": n_resamples,
        "valid_resamples": int(valid.sum()),
        "excluded_single_class_resamples": int((~valid).sum()),
        "seed": seed,
    }
