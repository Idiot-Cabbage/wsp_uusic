import torch
from torch import nn
from typing import Any

from .samus import Samus
from .moe_layers import MoEAdapter
from .common import Adapter


class MoESamus(nn.Module):
    """Wrapper around Samus adding MoE capabilities."""

    def __init__(self, samus: Samus, moe_expert_num: int = 4, top_k: int = 1, router_temp: float = 1.0):
        super().__init__()
        self.samus = samus
        self.moe_modules = []
        self._replace_adapters(self.samus.image_encoder, moe_expert_num, top_k, router_temp)

    def _replace_adapters(self, module: nn.Module, num_experts: int, top_k: int, temp: float) -> None:
        for name, child in module.named_children():
            if isinstance(child, Adapter):
                ratio = child.D_fc1.out_features / child.D_fc1.in_features
                moe = MoEAdapter(
                    child.D_fc1.in_features,
                    mlp_ratio=ratio,
                    num_experts=num_experts,
                    top_k=top_k,
                    temperature=temp,
                    skip_connect=child.skip_connect,
                    drop_path=0.0,
                )
                moe.experts[0].load_state_dict(child.state_dict())
                with torch.no_grad():
                    moe.router.linear.weight.zero_()
                    moe.router.linear.bias.fill_(-10.0)
                    moe.router.linear.bias[0] = 10.0
                setattr(module, name, moe)
                self.moe_modules.append(moe)
            else:
                self._replace_adapters(child, num_experts, top_k, temp)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.samus(*args, **kwargs)

    def get_aux_loss(self) -> torch.Tensor:
        losses = [m.get_aux_loss() for m in self.moe_modules]
        if losses:
            return 0.1 * torch.stack(losses).sum()
        return torch.tensor(0.0, device=self.samus.device)
