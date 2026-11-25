"""Fast tensor-only checks for the wearable stress model."""

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import WearableStressModel
from pyhealth.trainer import Trainer


@pytest.fixture
def sample_dataset():
    samples = [
        {
            "patient_id": f"p{i}",
            "sensors": [[float(i)] * 24 for _ in range(5)],
            "context": [float(i)] * 22,
            "stress": i % 2,
        }
        for i in range(3)
    ]
    return create_sample_dataset(
        samples,
        {"sensors": "tensor", "context": "tensor"},
        {"stress": "binary"},
    )


def test_supervised_forward_and_gradients(sample_dataset):
    model = WearableStressModel(sample_dataset)
    batch = {
        "sensors": torch.randn(3, 5, 24),
        "context": torch.randn(3, 22),
        "stress": torch.tensor([0, 1, 0]),
    }
    result = model(**batch)
    assert result["logit"].shape == (3, 1)
    assert result["y_prob"].shape == (3, 1)
    assert result["y_true"].shape == (3, 1)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.sensor_encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.context_encoder[0].weight.grad is not None
    assert model.supervised_classifier[1].weight.grad is not None


def test_pretrain_then_finetune(sample_dataset):
    model = WearableStressModel(sample_dataset, stage="pretrain")
    batch = {
        "sensors": torch.randn(3, 5, 24),
        "context": torch.randn(3, 22),
    }
    with pytest.raises(ValueError, match="text_emb"):
        model(**batch)
    result = model(**batch, text_emb=torch.randn(3, 768))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.text_projection.skip.weight.grad is not None
    model.set_stage("finetune")
    assert model(**batch, stress=torch.tensor([0, 1, 0]))["y_prob"].shape == (3, 1)


def test_input_validation(sample_dataset):
    with pytest.raises(ValueError, match="stage"):
        WearableStressModel(sample_dataset, stage="invalid")
    with pytest.raises(ValueError, match="contrastive_target"):
        WearableStressModel(sample_dataset, contrastive_target="invalid")
    model = WearableStressModel(sample_dataset)
    with pytest.raises(ValueError, match="sensors"):
        model(sensors=torch.randn(3, 24, 5), context=torch.randn(3, 22))


def test_pyhealth_trainer_one_step(sample_dataset):
    model = WearableStressModel(sample_dataset)
    loader = get_dataloader(sample_dataset, batch_size=3, shuffle=False)
    trainer = Trainer(model=model, enable_logging=False)
    trainer.train(train_dataloader=loader, epochs=1)
    result = model(**next(iter(loader)))
    assert result["y_prob"].shape == (3, 1)


def test_probe_freezes_weights_and_batchnorm(sample_dataset):
    model = WearableStressModel(sample_dataset, stage="probe")
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    loss = model(
        sensors=torch.randn(3, 5, 24),
        context=torch.randn(3, 22),
        stress=torch.tensor([0, 1, 0]),
    )["loss"]
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    for key, value in before.items():
        if not key.startswith("classifier."):
            assert torch.equal(value, after[key]), key
    assert not torch.equal(before["classifier.weight"], after["classifier.weight"])
    assert not model.context_encoder.training
    assert all(p.grad is None for p in model.sensor_encoder.parameters())
    model.set_stage("finetune")
    assert model.context_encoder.training
    assert all(p.requires_grad for p in model.sensor_encoder.parameters())


@pytest.mark.parametrize("target", ["reference", "stop_gradient", "one_hot"])
def test_checkpoint_roundtrip(sample_dataset, tmp_path, target):
    model = WearableStressModel(
        sample_dataset, stage="probe", dropout=0.2, contrastive_target=target
    ).eval()
    batch = {"sensors": torch.randn(3, 5, 24), "context": torch.randn(3, 22)}
    expected = model(**batch)["y_prob"]
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)
    restored = WearableStressModel.load_checkpoint(path, sample_dataset).eval()
    torch.testing.assert_close(restored(**batch)["y_prob"], expected)
    assert restored.stage == "probe"
    assert restored.model_config["dropout"] == 0.2
    assert restored.contrastive_target == target
    assert not restored.sensor_encoder.layers[0].self_attn.in_proj_weight.requires_grad


def test_pretraining_gradients_reach_both_views(sample_dataset):
    model = WearableStressModel(sample_dataset, stage="pretrain")
    result = model(
        sensors=torch.randn(3, 5, 24),
        context=torch.randn(3, 22),
        text_emb=torch.randn(3, 768),
    )
    result["loss"].backward()
    for parameter in [
        model.sensor_encoder.layers[0].self_attn.in_proj_weight,
        model.context_encoder[0].weight,
        model.sensor_projection.skip.weight,
        model.text_projection.skip.weight,
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.any(parameter.grad != 0)
    with pytest.raises(ValueError, match="at least 2"):
        model(
            sensors=torch.randn(1, 5, 24),
            context=torch.randn(1, 22),
            text_emb=torch.randn(1, 768),
        )


def test_target_objectives_match_explicit_loss_and_gradients(sample_dataset):
    """Equal soft-target forward losses must retain distinct gradient paths."""
    torch.manual_seed(7)
    model = WearableStressModel(sample_dataset, stage="pretrain", dropout=0).eval()
    batch = {
        "sensors": torch.randn(3, 5, 24),
        "context": torch.randn(3, 22),
        "text_emb": torch.randn(3, 768),
    }
    results = {}
    parameter = model.text_projection.skip.weight
    for mode in ("reference", "stop_gradient", "one_hot"):
        model.contrastive_target = mode
        actual = model(**batch)["loss"]
        sensor = model.encode(batch["sensors"], batch["context"])
        text = model.text_projection(batch["text_emb"])
        logits = text @ sensor.T / model.temperature
        if mode == "one_hot":
            expected = (
                -(
                    logits.log_softmax(1).diagonal().mean()
                    + logits.T.log_softmax(1).diagonal().mean()
                )
                / 2
            )
        else:
            target = (
                (sensor @ sensor.T + text @ text.T) * model.temperature / 2
            ).softmax(1)
            if mode == "stop_gradient":
                target = target.detach()
            expected = (
                -(
                    (target * logits.log_softmax(1)).sum(1).mean()
                    + (target.T * logits.T.log_softmax(1)).sum(1).mean()
                )
                / 2
            )
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad(actual, parameter)[0]
        expected_gradient = torch.autograd.grad(expected, parameter)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.isfinite(actual_gradient).all()
        results[mode] = (actual.detach(), actual_gradient)
    torch.testing.assert_close(results["reference"][0], results["stop_gradient"][0])
    assert not torch.allclose(results["reference"][1], results["stop_gradient"][1])


def test_legacy_checkpoint_keeps_detached_objective(sample_dataset, tmp_path):
    model = WearableStressModel(sample_dataset, contrastive_target="stop_gradient")
    legacy_config = dict(model.model_config)
    del legacy_config["contrastive_target"]
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": legacy_config,
            "stage": "pretrain",
        },
        path,
    )
    restored = WearableStressModel.load_checkpoint(path, sample_dataset)
    assert restored.contrastive_target == "stop_gradient"
