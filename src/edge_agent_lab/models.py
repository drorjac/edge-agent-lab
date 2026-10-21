"""Small convolutional sensor fusion models with deployable normalization."""

from __future__ import annotations

import torch
from torch import nn


class SensorNet(nn.Module):
    def __init__(self, width: int = 16, fusion: bool = False, accel_only: bool = False):
        super().__init__()
        self.width, self.fusion, self.accel_only = width, fusion, accel_only
        self.register_buffer("mean", torch.zeros(1, 9, 1))
        self.register_buffer("scale", torch.ones(1, 9, 1))
        if fusion:
            self.streams = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(3, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU()
                    )
                    for _ in range(3)
                ]
            )
            channels = width * 3
        else:
            channels = 3 if accel_only else 9
            self.streams = None
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, width * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),
            nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 16, width * 4), nn.ReLU(), nn.Dropout(0.2), nn.Linear(width * 4, 6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.scale
        if self.streams is not None:
            x = torch.cat(
                [stream(x[:, i * 3 : (i + 1) * 3]) for i, stream in enumerate(self.streams)], dim=1
            )
        elif self.accel_only:
            x = x[:, :3]
        return self.head(self.encoder(x))

    def config(self) -> dict:
        return {"width": self.width, "fusion": self.fusion, "accel_only": self.accel_only}


def _encode(value):
    if isinstance(value, torch.Tensor) and value.is_quantized:
        if value.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            return {
                "__qtensor__": "per_tensor",
                "data": value.int_repr(),
                "scale": value.q_scale(),
                "zero": value.q_zero_point(),
            }
        return {
            "__qtensor__": "per_channel",
            "data": value.int_repr(),
            "scales": value.q_per_channel_scales(),
            "zeros": value.q_per_channel_zero_points(),
            "axis": value.q_per_channel_axis(),
        }
    if isinstance(value, torch.dtype):
        if value != torch.qint8:
            raise ValueError(f"Unsupported packed dtype: {value}")
        return {"__dtype__": "qint8"}
    if isinstance(value, dict):
        import copy

        encoded = copy.copy(value)
        encoded.update({k: _encode(v) for k, v in value.items()})
        return encoded
    if isinstance(value, tuple):
        return tuple(_encode(v) for v in value)
    return value


def _decode(value):
    if isinstance(value, dict):
        if value.get("__qtensor__") == "per_tensor":
            return torch._make_per_tensor_quantized_tensor(
                value["data"], value["scale"], value["zero"]
            )
        if value.get("__qtensor__") == "per_channel":
            return torch._make_per_channel_quantized_tensor(
                value["data"], value["scales"], value["zeros"], value["axis"]
            )
        if value.get("__dtype__") == "qint8":
            return torch.qint8
        import copy

        decoded = copy.copy(value)
        decoded.update({k: _decode(v) for k, v in value.items()})
        return decoded
    if isinstance(value, tuple):
        return tuple(_decode(v) for v in value)
    return value


def load_model(path, quantized: bool = False) -> SensorNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SensorNet(**checkpoint["config"]).eval()
    if quantized:
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(_decode(checkpoint["state"]))
    return model


def save_model(model, path, config: dict) -> None:
    """Pack quantized weights as integer tensors and primitive quantization metadata.

    PyTorch 2.2's restricted loader rejects native quantized tensor rebuild globals.
    This format keeps weights_only=True for both floating and quantized checkpoints.
    """
    torch.save({"config": config, "state": _encode(model.state_dict())}, path)
