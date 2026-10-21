"""Fixed reference ESP32-P4 DS-TC-ResNet, with six external window states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from device_contract import validate_contract, STATE_SHAPES


class DSTCBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, repeat: int, kernel_size: int,
        stride: int, dilation: int, separable: bool, residual: bool,
        padding: str, pool: int, dropout: float,
    ):
        super().__init__()
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        self.context = dilation * (kernel_size - 1)
        self.padding = padding
        self.stride = stride
        self.pool_size = pool
        self.residual_enabled = residual
        layers = []
        current_channels = in_channels
        for _ in range(repeat):
            groups = current_channels if separable else 1
            temporal = nn.Conv1d(
                current_channels, current_channels if separable else out_channels,
                kernel_size, stride=stride, dilation=dilation, groups=groups,
                bias=False,
            )
            pointwise = nn.Conv1d(current_channels, out_channels, 1, bias=False) if separable else nn.Identity()
            layers.append(nn.ModuleDict({
                "temporal": temporal,
                "pointwise": pointwise,
                "norm": nn.BatchNorm1d(out_channels),
            }))
            current_channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if residual else None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(pool, stride=pool) if pool > 1 else nn.Identity()

    def _pad(self, x: torch.Tensor, context: int) -> torch.Tensor:
        if self.padding == "causal":
            return F.pad(x, (context, 0))
        if self.padding == "same":
            left = context // 2
            return F.pad(x, (left, context - left))
        raise ValueError(f"Unsupported padding: {self.padding}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer["temporal"](self._pad(x, self.context))
            x = layer["pointwise"](x)
            x = layer["norm"](x)
            if self.residual is None:
                x = self.dropout(self.activation(x))
        if self.residual is not None:
            x = x + self.residual(residual)
            x = self.dropout(self.activation(x))
        return self.pool(x)

    def stream(self, x: torch.Tensor, states: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.padding != "causal" or self.stride != 1 or self.pool_size != 1:
            raise ValueError("Streaming requires causal padding, stride=1, and pool=1")
        next_states = []
        residual = x
        state_iterator = iter(states)
        for layer in self.layers:
            state = next(state_iterator) if self.context else None
            history = state[:, 1:, 0, :].transpose(1, 2) if state is not None else None
            combined = torch.cat((history, x), dim=2) if state is not None else x
            x = layer["temporal"](combined)
            if self.context:
                next_states.append(combined.transpose(1, 2).unsqueeze(2))
            x = layer["pointwise"](x)
            x = layer["norm"](x)
            if self.residual is None:
                x = self.dropout(self.activation(x))
        if self.residual is not None:
            x = self.dropout(self.activation(x + self.residual(residual)))
        return x, next_states


@dataclass(frozen=True)
class StateSpec:
    channels: int
    context: int


class DSTCResNet(nn.Module):
    def __init__(self, config: dict, label_count: int):
        super().__init__()
        validate_contract(config)
        if not 3 <= label_count <= 34:
            raise ValueError("ESP32-P4 requires 3 to 34 classes")
        feature_channels = int(config["features"]["mfcc_bins"])
        model_config = config["model"]
        blocks = []
        in_channels = feature_channels
        for values in zip(
            model_config["filters"], model_config["repeats"], model_config["kernel_sizes"],
            model_config["strides"], model_config["dilations"], model_config["separable"],
            model_config["residual"], model_config["padding"], model_config["pools"],
        ):
            out_channels, repeat, kernel, stride, dilation, separable, residual, padding, pool = values
            block = DSTCBlock(
                in_channels, out_channels, repeat, kernel, stride, dilation,
                separable, residual, padding, pool, model_config["dropout"],
            )
            blocks.append(block)
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(in_channels, label_count)
        self.output_channels = in_channels
        self.pooling_frames = 98

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        for block in self.blocks:
            x = block(x)
        x = x[:, :, -self.pooling_frames:]
        if x.shape[2] < self.pooling_frames:
            x = F.pad(x, (self.pooling_frames - x.shape[2], 0))
        return self.classifier(x.mean(dim=2))

    def state_specs(self) -> list[StateSpec]:
        """Full stored window lengths, including the current frame."""
        return [StateSpec(shape[2], shape[0]) for shape in STATE_SHAPES]

    def initial_stream_state(self, batch_size: int, device: torch.device | str) -> list[torch.Tensor]:
        return [torch.zeros(batch_size, *shape, device=device) for shape in STATE_SHAPES]

    def stream_step(
        self, features: torch.Tensor, states: list[torch.Tensor], *, return_unused=False,
    ):
        if features.shape[2] != 1 or len(states) != 6:
            raise ValueError("Fixed ESP32-P4 streaming requires one frame and six states")
        x = features
        next_states = []
        offset = 0
        for block in self.blocks:
            unused = x
            state_count = len(block.layers) if block.context else 0
            x, block_states = block.stream(x, states[offset:offset + state_count])
            next_states.extend(block_states)
            offset += state_count
        history = states[-1][:, 1:, 0, :].transpose(1, 2)
        window = torch.cat((history, x), dim=2)
        next_states.append(window.transpose(1, 2).unsqueeze(2))
        logits = self.classifier(window.mean(dim=2))
        if return_unused:
            return logits, next_states, unused.transpose(1, 2).unsqueeze(2)
        return logits, next_states


class InternalStreamingRunner:
    """State-owning runtime facade for internal-state streaming inference."""

    def __init__(self, model: DSTCResNet, batch_size: int = 1):
        self.model = model
        self.batch_size = batch_size
        self.reset()

    def reset(self) -> None:
        device = next(self.model.parameters()).device
        self.states = self.model.initial_stream_state(self.batch_size, device)

    @torch.no_grad()
    def process(self, features: torch.Tensor) -> torch.Tensor:
        logits, self.states = self.model.stream_step(features, self.states)
        return logits
