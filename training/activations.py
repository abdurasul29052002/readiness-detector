"""
Custom aktivatsiya funksiyalari va ResNet50 ichidagi ReLU larni
almashtirish uchun yordamchi funksiyalar.

Ilmiy yangilik: LearnableSwish — Swish (SiLU) ning β parametri
har bir ResNet blokida alohida o'rganiluvchi qilingan variant.
Disbalansli sinflar ustida modelga aktivatsiya shaklini
ma'lumotdan o'rganish imkoniyatini beradi.

Formula:
  f(x) = x * sigmoid(beta * x)
  beta = nn.Parameter (init=1.0, scope: per-block | per-channel | global)

Foydalanish:
  from activations import build_activation, replace_activations
  factory = build_activation("lswish")           # per-block (default)
  factory = build_activation("lswish_channel", channels=64)
  replace_activations(model, factory)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class LearnableSwish(nn.Module):
    """f(x) = x * sigmoid(beta * x), beta — o'rganiluvchi skalyar.

    Per-block scope: bitta ResNet Bottleneck moduli ichidagi 3 ta
    chaqiriqda bir xil beta ishlatiladi (chunki torchvision Bottleneck
    self.relu ni qayta foydalanadi)."""

    def __init__(self, beta_init: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

    def extra_repr(self) -> str:
        return f"beta={self.beta.item():.4f}"


class LearnableSwishChannel(nn.Module):
    """Per-channel beta. Channels — kirish tensori kanallari soni.
    PReLU bilan o'xshash, lekin Swish formulasida."""

    def __init__(self, num_channels: int, beta_init: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        self.beta = nn.Parameter(torch.full((num_channels,), float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W] yoki [B, C]
        if x.dim() == 4:
            beta = self.beta.view(1, -1, 1, 1)
        elif x.dim() == 2:
            beta = self.beta.view(1, -1)
        else:
            beta = self.beta
        return x * torch.sigmoid(beta * x)

    def extra_repr(self) -> str:
        return f"channels={self.num_channels}, beta_mean={self.beta.mean().item():.4f}"


# ──────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────
ActivationFactory = Callable[[int | None], nn.Module]


def build_activation(name: str) -> ActivationFactory:
    """Aktivatsiya factory qaytaradi.

    Factory(channels) -> nn.Module. channels argumenti faqat
    per-channel variantlar uchun ishlatiladi.
    """
    name = name.lower()

    if name == "relu":
        return lambda channels=None: nn.ReLU(inplace=True)
    if name == "mish":
        return lambda channels=None: nn.Mish(inplace=False)
    if name == "silu" or name == "swish":
        return lambda channels=None: nn.SiLU(inplace=False)
    if name == "gelu":
        return lambda channels=None: nn.GELU()
    if name == "lswish":
        return lambda channels=None: LearnableSwish(beta_init=1.0)
    if name == "lswish_channel":
        def _f(channels):
            if channels is None:
                raise ValueError("lswish_channel uchun channels kerak")
            return LearnableSwishChannel(channels, beta_init=1.0)
        return _f
    raise ValueError(f"Noma'lum aktivatsiya: {name}")


# ──────────────────────────────────────────────────────────────
# REPLACE
# ──────────────────────────────────────────────────────────────
def replace_activations(module: nn.Module, factory: ActivationFactory,
                        target=nn.ReLU) -> int:
    """Modul ichidagi barcha `target` (default: nn.ReLU) modullarini
    factory() yordamida yaratilgan yangi modul bilan rekursiv almashtiradi.

    Per-channel factorylar uchun parent modul kanallarini aniqlashga
    urinadi (Conv2d.out_channels yoki BatchNorm2d.num_features).

    Qaytaradi: almashtirilgan modullar soni.
    """
    count = 0
    # parent modul ichida oldingi qatlamning kanal sonini topish
    prev_channels = None
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d,)):
            prev_channels = child.out_channels
        elif isinstance(child, (nn.BatchNorm2d, nn.GroupNorm)):
            prev_channels = getattr(child, "num_features",
                                    getattr(child, "num_channels", prev_channels))
        if isinstance(child, target):
            try:
                new_act = factory(prev_channels)
            except TypeError:
                new_act = factory()
            setattr(module, name, new_act)
            count += 1
        else:
            count += replace_activations(child, factory, target)
    return count


def count_activation_params(module: nn.Module) -> int:
    """LearnableSwish/Channel modullaridagi o'rganiluvchi parametrlar soni."""
    total = 0
    for m in module.modules():
        if isinstance(m, (LearnableSwish, LearnableSwishChannel)):
            for p in m.parameters():
                total += p.numel()
    return total


def list_beta_values(module: nn.Module) -> list[tuple[str, float]]:
    """Barcha LearnableSwish modullarining nomi va beta qiymatini qaytaradi
    (training tugagandan keyin tahlil uchun)."""
    out = []
    for name, m in module.named_modules():
        if isinstance(m, LearnableSwish):
            out.append((name, m.beta.item()))
        elif isinstance(m, LearnableSwishChannel):
            out.append((name, m.beta.mean().item()))
    return out
