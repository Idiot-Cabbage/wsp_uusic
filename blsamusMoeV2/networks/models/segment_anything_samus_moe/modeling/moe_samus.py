import torch
from torch import nn
from typing import Any, Tuple

from .samus import Samus
from .moe_layers import MoEAdapter


class MoESamus(nn.Module):
    """Wrapper around Samus adding MoE capabilities."""

    def __init__(self, samus: Samus, moe_expert_num: int = 4, top_k: int = 1, router_temp: float = 1.0):
        super().__init__()
        self.samus = samus
        self.moe_modules = []
        self._convert(moe_expert_num, top_k, router_temp)

    def _convert(self, num_experts: int, top_k: int, temp: float) -> None:
        enc = self.samus.image_encoder
        base_adp = enc.input_Adapter
        enc.input_Adapter = MoEAdapter(
            base_adp.D_fc1.in_features,
            mlp_ratio=base_adp.D_fc1.out_features / base_adp.D_fc1.in_features,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temp,
            skip_connect=True,
            drop_path=0.0,
        )
        enc.input_Adapter.experts[0].load_state_dict(base_adp.state_dict())
        with torch.no_grad():
            enc.input_Adapter.router.linear.weight.data.zero_()
            enc.input_Adapter.router.linear.bias.data.fill_(-10.0)
            enc.input_Adapter.router.linear.bias.data[0] = 10.0
        self.moe_modules.append(enc.input_Adapter)
        total = len(enc.blocks)
        for i, blk in enumerate(enc.blocks):
            if blk.window_size == 0:
                ar = blk.MLP_Adapter
                ratio_a = ar.D_fc1.out_features / ar.D_fc1.in_features
                moe_adp = MoEAdapter(
                    ar.D_fc1.in_features,
                    mlp_ratio=ratio_a,
                    num_experts=num_experts,
                    top_k=top_k,
                    temperature=temp,
                    skip_connect=False,
                    drop_path=0.2 if i >= total // 2 else 0.0,
                )
                moe_adp.experts[0].load_state_dict(ar.state_dict())
                with torch.no_grad():
                    moe_adp.router.linear.weight.data.zero_()
                    moe_adp.router.linear.bias.data.fill_(-10.0)
                    moe_adp.router.linear.bias.data[0] = 10.0
                blk.MLP_Adapter = moe_adp
                self.moe_modules.append(moe_adp)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.samus(*args, **kwargs)

    def get_aux_loss(self) -> torch.Tensor:
        losses = [m.get_aux_loss() for m in self.moe_modules]
        if losses:
            return 0.1 * torch.stack(losses).sum()
        return torch.tensor(0.0, device=self.samus.device)
