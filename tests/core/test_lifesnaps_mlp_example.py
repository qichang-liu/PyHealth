"""Protect the example's held-out preprocessing and metric edge cases."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = (
    Path(__file__).parents[2] / "examples/wearable/lifesnaps_stressdetection_mlp.py"
)
SPEC = importlib.util.spec_from_file_location("lifesnaps_mlp_example", SCRIPT)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class TinySamples:
    """Minimal in-memory samples with the dataloader's shuffle interface."""

    def __init__(self, values):
        self.rows = [
            {
                "sensors": torch.tensor([value], dtype=torch.float32),
                "context": torch.tensor([7.0]),
                "stress": torch.tensor([index % 2]),
            }
            for index, value in enumerate(values)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle


def test_normalization_uses_training_statistics_only():
    train, validation, test = (TinySamples(x) for x in ([1, 3], [5], [101]))
    loaders = example._normalized_loaders((train, validation, test))
    # Training mean=2, population std=1. Held-out values must not change them.
    torch.testing.assert_close(next(iter(loaders[1]))["sensors"], torch.tensor([[3.0]]))
    torch.testing.assert_close(
        next(iter(loaders[2]))["sensors"], torch.tensor([[99.0]])
    )
    # Constant context stays finite and centered, with scale falling to 1.
    torch.testing.assert_close(next(iter(loaders[2]))["context"], torch.tensor([[0.0]]))
    assert train.shuffle is True
    assert validation.shuffle is False
    assert test.shuffle is False


def test_empty_partition_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        example._normalized_loaders(
            (TinySamples([1, 3]), TinySamples([]), TinySamples([5]))
        )


def test_probability_auc_and_single_class_reporting():
    result = example._binary_metrics([0, 1], [0.2, 0.9], 0.1)
    assert result["test_roc_auc"] == 1.0
    result = example._binary_metrics([0, 0], [0.2, 0.9], 0.1)
    assert result["test_roc_auc"] is None
    assert result["test_roc_auc_reason"] == "single-class test partition"


def test_invalid_predictions_are_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        example._binary_metrics([0, 1], [0.2, np.nan], 0.1)
    with pytest.raises(ValueError, match="matching"):
        example._binary_metrics([0, 1], [0.2], 0.1)
