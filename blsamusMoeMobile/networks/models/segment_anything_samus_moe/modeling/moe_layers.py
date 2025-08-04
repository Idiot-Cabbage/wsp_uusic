import torch
from torch import nn
import torch.nn.functional as F

from .common import MLPBlock, Adapter


class DropPath(nn.Module):
    """Stochastic depth as in timm."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random.floor_()
        return x.div(keep) * random


class Router(nn.Module):
    """Gating mechanism for MoE."""

    def __init__(self, dim: int, num_experts: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, num_experts)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x) / self.temperature
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.5
        return logits


class MoEFFN(nn.Module):
    """Feedforward network with Mixture of Experts."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 4,
        top_k: int = 1,
        temperature: float = 1.0,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [MLPBlock(embedding_dim=dim, mlp_dim=hidden, act=act_layer)
             for _ in range(num_experts)]
        )
        self.router = Router(dim, num_experts, temperature)
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.last_routing = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        if self.training:
            idx = torch.multinomial(probs.view(-1, self.num_experts), 1)
            idx = idx.view(probs.shape[:-1])
        else:
            idx = probs.argmax(dim=-1)
        dispatch_mask = F.one_hot(idx, self.num_experts).type_as(probs)
        out = 0.0
        for i, expert in enumerate(self.experts):
            out = out + expert(x) * dispatch_mask[..., i : i + 1]
        self.last_routing = dispatch_mask.float().mean(dim=(0, 1)).detach()
        ideal = torch.full_like(self.last_routing, 1.0 / self.num_experts)
        self.aux_loss = F.mse_loss(self.last_routing, ideal)
        return out

    def get_aux_loss(self) -> torch.Tensor:
        return self.aux_loss


class MoEAdapter(nn.Module):
    """Adapter layer with Mixture of Experts."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 0.25,
        num_experts: int = 4,
        top_k: int = 1,
        temperature: float = 1.0,
        skip_connect: bool = True,
        act_layer: nn.Module = nn.GELU,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.skip_connect = skip_connect
        self.experts = nn.ModuleList(
            [Adapter(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, skip_connect=False)
             for _ in range(num_experts)]
        )
        self.router = Router(dim, num_experts, temperature)
        self.dropout = nn.Dropout(0.1)
        self.drop_path = DropPath(drop_path)
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.last_routing = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        if self.training:
            idx = torch.multinomial(probs.view(-1, self.num_experts), 1)
            idx = idx.view(probs.shape[:-1])
        else:
            idx = probs.argmax(dim=-1)
        dispatch_mask = F.one_hot(idx, self.num_experts).type_as(probs)
        out = 0.0
        for i, expert in enumerate(self.experts):
            out = out + expert(x) * dispatch_mask[..., i : i + 1]
        out = self.dropout(out)
        if self.skip_connect:
            out = out + x
        out = self.drop_path(out)
        self.last_routing = dispatch_mask.float().mean(dim=(0, 1)).detach()
        ideal = torch.full_like(self.last_routing, 1.0 / self.num_experts)
        self.aux_loss = F.mse_loss(self.last_routing, ideal)
        return out

    def get_aux_loss(self) -> torch.Tensor:
        return self.aux_loss
