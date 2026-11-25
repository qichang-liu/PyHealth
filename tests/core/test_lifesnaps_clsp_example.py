"""Small deterministic checks of example preprocessing and temporal controls."""

import importlib.util
from pathlib import Path

import numpy as np


def example_helpers():
    path = Path(__file__).parents[2] / "examples/wearable/lifesnaps_clsp_utils.py"
    spec = importlib.util.spec_from_file_location("lifesnaps_clsp_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calendar_gap_and_train_fallback_preserve_static():
    helper = example_helpers()
    context = np.arange(4 * 22, dtype=float).reshape(4, 22)
    ids = np.array(["p1", "p1", "p1", "p2"])
    dates = np.array(["2021-06-01", "2021-06-04", "2021-06-10", "2021-06-07"])
    fallback = np.full(22, -7.0)
    result, count = helper.calendar_context(
        context, ids, dates, np.array([0, 2]), 3, fallback
    )
    dynamic = helper.DYNAMIC_INDICES
    static = [i for i in range(22) if i not in dynamic]
    assert count == 1
    np.testing.assert_array_equal(result[0, dynamic], fallback[dynamic])
    # June 10 minus three calendar days selects p1 June 4, not p2 June 7.
    np.testing.assert_array_equal(result[1, dynamic], context[1, dynamic])
    np.testing.assert_array_equal(result[:, static], context[[0, 2]][:, static])
    unchanged, count = helper.calendar_context(
        context, ids, dates, np.array([0, 2]), 0, fallback
    )
    np.testing.assert_array_equal(unchanged, context[[0, 2]])
    assert count == 0


def test_preprocessing_roundtrip_does_not_fit_test_values():
    helper = example_helpers()
    sensors = np.arange(3 * 5 * 24, dtype=float).reshape(3, 5, 24)
    context = np.arange(3 * 22, dtype=float).reshape(3, 22)
    normalizer = helper.TrainOnlyNormalizer().fit(sensors[:2], context[:2])
    state = normalizer.to_dict()
    expected = normalizer.transform(sensors, context)
    assert normalizer.to_dict() == state
    assert np.max(normalizer.sensor.quantiles_) == np.max(sensors[:2])
    restored = helper.TrainOnlyNormalizer.from_dict(state)
    for a, b in zip(expected, restored.transform(sensors, context)):
        np.testing.assert_array_equal(a, b)


def test_participant_split_is_disjoint_and_complete():
    helper = example_helpers()
    ids = np.array(["p1", "p1", "p2", "p3", "p4"])
    split = helper.split_participants(ids, 0)
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(
        range(len(ids))
    )
    groups = [set(ids[rows]) for rows in split.values()]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1 :])


def test_text_statistics_are_finite_for_constant_channels():
    helper = example_helpers()
    text = helper.build_text(np.ones((5, 24)), "gender: female")
    assert "gender: female" in text
    assert "kurtosis 0.00" in text
    assert "nan" not in text and "stai" not in text


def test_direct_statistics_match_prompt_and_handle_constant_values():
    helper = example_helpers()
    features = helper.statistical_features(np.ones((2, 5, 24)), np.ones((2, 22)))
    assert features.shape == (2, 77)
    np.testing.assert_array_equal(features[0, :11], [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, -22:], 1)
