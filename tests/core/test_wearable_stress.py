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


def test_checkpoint_roundtrip(sample_dataset, tmp_path):
    model = WearableStressModel(sample_dataset, stage="probe", dropout=0.2).eval()
    batch = {"sensors": torch.randn(3, 5, 24), "context": torch.randn(3, 22)}
    expected = model(**batch)["y_prob"]
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)
    restored = WearableStressModel.load_checkpoint(path, sample_dataset).eval()
    torch.testing.assert_close(restored(**batch)["y_prob"], expected)
    assert restored.stage == "probe"
    assert restored.model_config["dropout"] == 0.2
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
