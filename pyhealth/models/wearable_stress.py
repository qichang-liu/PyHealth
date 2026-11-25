"""Wearable sensor and context model for daily stress prediction.

Authors: Xiangwen Wang (xw120@illinois.edu), Qichang Liu (qichang4@illinois.edu).
Paper: https://proceedings.mlr.press/v287/yang25a.html

Independent PyHealth implementation based on Yang et al., CHIL 2025, and
documented reference-code behavior. No author source is imported here.
The optional contrastive stage accepts externally computed text embeddings;
this module does not download or train BERT.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class _ResidualConv(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.BatchNorm1d(width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.layers(x))


class _Projection(nn.Module):
    def __init__(self, input_dim: int, dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.skip = nn.Linear(input_dim, dim)
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        return self.norm(skip + self.dropout(self.fc(F.gelu(skip))))


class WearableStressModel(BaseModel):
    """Transformer/ResNet fusion with optional text contrastive pretraining.

    Sensor channels are attention tokens; each token holds 24 hourly values.
    ``stage='supervised'`` trains the fused encoder from scratch.
    ``stage='pretrain'`` aligns it with externally supplied 768-D BERT vectors.
    ``stage='probe'`` freezes the representation, including BatchNorm statistics.
    ``stage='finetune'`` trains a binary classifier after pretraining.

    By default, soft targets remain differentiable as in the reference code.
    Detached soft targets and one-hot paired targets are explicit ablations.
    Matching the objective does not establish equivalence of the full pipeline.

    Args:
        dataset: SampleDataset with ``sensors``, ``context``, and ``stress``.
        stage: ``supervised``, ``pretrain``, ``probe``, or ``finetune``.
        channels: number of sensor channels, five for LifeSnaps.
        hours: number of hourly values per channel.
        context_dim: number of context fields, 22 for the current task.
        temperature: contrastive logit scale denominator.
        dropout: Transformer and projection dropout probability.
        transformer_layers: Number of channel-attention layers.
        contrastive_target: ``reference`` keeps soft-target gradients;
            ``stop_gradient`` detaches them; ``one_hot`` uses paired indices.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": "p0", "sensors": [[0.] * 24] * 5,
        ...             "context": [0.] * 22, "stress": 0}]
        >>> data = create_sample_dataset(samples, {"sensors": "tensor",
        ...     "context": "tensor"}, {"stress": "binary"})
        >>> model = WearableStressModel(data)
        >>> model(sensors=torch.zeros(1, 5, 24),
        ...       context=torch.zeros(1, 22))["y_prob"].shape
        torch.Size([1, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        stage: str = "supervised",
        channels: int = 5,
        hours: int = 24,
        context_dim: int = 22,
        temperature: float = 0.07,
        dropout: float = 0.1,
        transformer_layers: int = 3,
        contrastive_target: str = "reference",
    ) -> None:
        super().__init__(dataset)
        if stage not in {"supervised", "pretrain", "probe", "finetune"}:
            raise ValueError(f"Unknown stage: {stage}")
        if self.mode != "binary" or self.feature_keys != ["sensors", "context"]:
            raise ValueError("Expected sensors/context inputs and one binary label")
        if self.label_keys != ["stress"]:
            raise ValueError("Expected the stress label")
        if channels < 1 or hours < 1 or context_dim < 1:
            raise ValueError("Input dimensions must be positive")
        if hours % 4 or temperature <= 0:
            raise ValueError("Hours must be divisible by 4; temperature > 0")
        if not 0 <= dropout < 1 or transformer_layers < 1:
            raise ValueError("dropout must be in [0, 1); transformer_layers >= 1")
        if contrastive_target not in {"reference", "stop_gradient", "one_hot"}:
            raise ValueError("Unknown contrastive_target")

        self.stage = stage
        self.channels = channels
        self.hours = hours
        self.context_dim = context_dim
        self.temperature = temperature
        self.contrastive_target = contrastive_target
        self.model_config = {
            "channels": channels,
            "hours": hours,
            "context_dim": context_dim,
            "temperature": temperature,
            "dropout": dropout,
            "transformer_layers": transformer_layers,
            "contrastive_target": contrastive_target,
        }
        layer = nn.TransformerEncoderLayer(
            d_model=hours,
            nhead=4,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
        )
        self.sensor_encoder = nn.TransformerEncoder(
            layer, num_layers=transformer_layers
        )
        self.context_encoder = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.ReLU(),
            *[_ResidualConv(128) for _ in range(3)],
        )
        self.context_head = nn.Linear(128, 64)
        self.fusion = nn.Linear(hours + 64, 32)
        self.sensor_projection = _Projection(32, dropout=dropout)
        self.text_projection = _Projection(768, dropout=dropout)
        self.supervised_classifier = nn.Sequential(nn.ReLU(), nn.Linear(32, 1))
        self.classifier = nn.Linear(128, 1)
        self.set_stage(stage)

    def set_stage(self, stage: str) -> None:
        """Select a training stage without discarding learned weights.

        Args:
            stage: One of supervised, pretrain, probe, or finetune.

        Raises:
            ValueError: If stage is unsupported.
        """
        if stage not in {"supervised", "pretrain", "probe", "finetune"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(stage != "probe" or name.startswith("classifier."))
        self.train(self.training)

    def train(self, mode: bool = True) -> "WearableStressModel":
        """Set training mode while keeping the linear-probe encoder frozen.

        Args:
            mode: Whether trainable modules should use training behavior.

        Returns:
            This model, following the PyTorch train/eval convention.
        """
        super().train(mode)
        if self.stage == "probe":
            for module in (
                self.sensor_encoder,
                self.context_encoder,
                self.context_head,
                self.fusion,
                self.sensor_projection,
                self.text_projection,
            ):
                module.eval()
        return self

    def save_checkpoint(self, path: str | Path) -> None:
        """Save weights, architecture configuration, and active stage.

        Args:
            path: Destination checkpoint file; its parent must exist.

        Raises:
            OSError: If the destination cannot be written.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.model_config,
                "stage": self.stage,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, dataset: SampleDataset
    ) -> "WearableStressModel":
        """Restore a checkpoint on CPU in the constructor's training mode.

        Args:
            path: File created by save_checkpoint.
            dataset: Compatible supervised SampleDataset describing inputs.

        Returns:
            Restored model. Call eval() before deterministic inference.

        Raises:
            RuntimeError: If checkpoint weights do not match the architecture.
            ValueError: If its configuration or dataset schema is incompatible.
        """
        payload: dict[str, Any] = torch.load(
            path, map_location="cpu", weights_only=True
        )
        config = dict(payload["config"])
        # Older checkpoints predate this option and used detached soft targets.
        config.setdefault("contrastive_target", "stop_gradient")
        model = cls(dataset, stage=payload["stage"], **config)
        model.load_state_dict(payload["state_dict"])
        return model

    def fuse(self, sensors: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Encode and fuse the two numerical input modalities.

        Args:
            sensors: Tensor shaped (batch, channels, hours).
            context: Tensor shaped (batch, context_dim).

        Returns:
            Fused tensor shaped (batch, 32), on the model's device.

        Raises:
            ValueError: If input dimensions or batch sizes are incompatible.
        """
        if sensors.ndim != 3 or sensors.shape[1:] != (self.channels, self.hours):
            raise ValueError("sensors must have shape (batch, channels, hours)")
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape (batch, context_dim)")
        if sensors.shape[0] != context.shape[0]:
            raise ValueError("sensors and context batch sizes differ")
        sensor = self.sensor_encoder(sensors.to(self.device).float()).mean(dim=1)
        tab = self.context_encoder(context.to(self.device).float().unsqueeze(1))
        tab = self.context_head(tab.mean(dim=-1))
        return self.fusion(torch.cat((sensor, tab), 1))

    def encode(self, sensors: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Project fused sensor and context inputs.

        Args:
            sensors: Tensor shaped (batch, channels, hours).
            context: Tensor shaped (batch, context_dim).

        Returns:
            Multimodal representation shaped (batch, 128).
        """
        return self.sensor_projection(self.fuse(sensors, context))

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute stage-specific predictions or contrastive loss.

        Args:
            **kwargs: sensors and context tensors; pretrain also requires
                text_emb shaped (batch, 768). Classification may include
                binary stress labels shaped (batch,) or (batch, 1).

        Returns:
            Pretrain: scalar loss and (batch, 128) embed. Classification:
            (batch, 1) logit/y_prob, plus loss/y_true when stress is supplied.

        Raises:
            ValueError: If feature/text shapes or contrastive batch size fail
                validation, or pretrain has no paired text_emb.
            KeyError: If sensors or context is missing.
        """
        fused = self.fuse(kwargs["sensors"], kwargs["context"])
        if self.stage == "pretrain":
            if fused.shape[0] < 2:
                raise ValueError("Contrastive pretraining requires at least 2 pairs")
            embedding = self.sensor_projection(fused)
            if "text_emb" not in kwargs:
                raise ValueError("Pretraining requires paired text_emb")
            text_emb = kwargs["text_emb"].to(self.device).float()
            if text_emb.shape != (embedding.shape[0], 768):
                raise ValueError("text_emb must have shape (batch, 768)")
            text = self.text_projection(text_emb)
            logits = text @ embedding.T / self.temperature
            if self.contrastive_target == "one_hot":
                targets = torch.arange(len(embedding), device=embedding.device)
                reverse_targets = targets
            else:
                targets = F.softmax(
                    (embedding @ embedding.T + text @ text.T) * self.temperature / 2,
                    dim=-1,
                )
                if self.contrastive_target == "stop_gradient":
                    targets = targets.detach()
                reverse_targets = targets.T
            loss = (
                F.cross_entropy(logits, targets)
                + F.cross_entropy(logits.T, reverse_targets)
            ) / 2
            return {"loss": loss, "embed": embedding}

        if self.stage == "supervised":
            logits = self.supervised_classifier(fused)
        else:
            logits = self.classifier(self.sensor_projection(fused))
        result = {"logit": logits, "y_prob": self.prepare_y_prob(logits)}
        if "stress" in kwargs:
            y_true = kwargs["stress"].to(self.device).float().reshape(-1, 1)
            result["loss"] = F.binary_cross_entropy_with_logits(logits, y_true)
            result["y_true"] = y_true
        return result
