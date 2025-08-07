import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Optional
from .common import LayerNorm2d

# 添加输出投影层，确保输出维度正确
class OutputProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.projection(x)

# 修改 MoE 结构
class MOEConfig:
    def __init__(
        self, 
        hidden_dim, 
        output_dim,  # 添加输出维度参数
        expert_number, 
        top_k, 
        shared_experts_number=2,
    ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # 新增输出维度
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    
    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)
        
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        
        return router_logits, router_weights, selected_experts, expert_mask

class SparseMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim  # 使用配置中的输出维度
        self.expert_number = config.expert_number
        self.top_k = config.top_k

        # 专家输出指定维度
        self.experts = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.output_dim) for _ in range(self.expert_number)]
        )
        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)
        
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, self.output_dim),  # 使用配置的输出维度
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.numel() > 0:
                current_state = hidden_states[top_x, :]
                current_hidden_states = expert_layer(
                    current_state
                ) * router_weights[top_x, idx].unsqueeze(-1)
                
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, self.output_dim)
        return final_hidden_states, router_logits

class ShareExpertMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.moe_model = SparseMOE(config)
        # 共享专家也使用配置的输出维度
        self.shared_experts = nn.ModuleList(
            [
                nn.Linear(
                    config.hidden_dim, config.output_dim
                ) for _ in range(config.shared_experts_number)
            ]
        )

    def forward(self, x):
        sparse_moe_out, router_logits = self.moe_model(x)
        
        shared_experts_out = torch.stack(
            [expert(x) for expert in self.shared_experts], dim=0
        ).sum(dim=0)
        
        return sparse_moe_out + shared_experts_out, router_logits

# 修复 MaskDecoder
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        moe_experts: int = 4,
        moe_top_k: int = 2,
        shared_experts: int = 2
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 使用修正后的MoE配置
        moe_config = MOEConfig(
            hidden_dim=transformer_dim,
            output_dim=transformer_dim // 8,  # 确保输出维度正确
            expert_number=moe_experts,
            top_k=moe_top_k,
            shared_experts_number=shared_experts
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                ShareExpertMOE(moe_config)
                for _ in range(self.num_mask_tokens)
            ]
        )

        # IOU预测头使用单独配置
        iou_moe_config = MOEConfig(
            hidden_dim=transformer_dim,
            output_dim=self.num_mask_tokens,  # 输出类别数
            expert_number=moe_experts,
            top_k=moe_top_k,
            shared_experts_number=shared_experts
        )
        self.iou_prediction_head = ShareExpertMOE(iou_moe_config)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks, iou_pred, router_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
            
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        
        return masks, iou_pred, router_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if len(image_embeddings.shape) == 3:
            image_embeddings = image_embeddings.unsqueeze(0)
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
            
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        b, c, h, w = upscaled_embedding.shape

        hyper_in_list = []
        router_logits_list = []
        
        # 修正后的MoE处理
        for i in range(self.num_mask_tokens):
            moe_input = mask_tokens_out[:, i, :].unsqueeze(1)
            hyper_in, logits = self.output_hypernetworks_mlps[i](moe_input)
            hyper_in_list.append(hyper_in.squeeze(1))
            router_logits_list.append(logits)

        hyper_in = torch.stack(hyper_in_list, dim=1)
        # 维度修正后的矩阵乘法
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # IOU预测处理
        iou_input = iou_token_out.unsqueeze(1)
        iou_pred, iou_logits = self.iou_prediction_head(iou_input)
        iou_pred = iou_pred.squeeze(1)
        router_logits_list.append(iou_logits)

        return masks, iou_pred, router_logits_list