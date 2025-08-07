import torch
from utils.loss_functions.TopK_loss import TopKLoss
from utils.loss_functions.crossentropy import RobustCrossEntropyLoss
from utils.loss_functions.nd_softmax import softmax_helper
from utils.loss_functions.tensor_utils import sum_tensor
from torch import nn
import numpy as np

def switch_load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    计算 Switch Transformers 的负载均衡损失
    
    Args:
        router_logits: shape [batch_size * sequence_length, num_experts]
        num_experts: 专家数量
    
    Returns:
        total_loss: 总损失 = auxiliary_loss + z_loss
    """
    # 计算路由概率
    router_probs = torch.softmax(router_logits, dim=-1)  # [b*s, num_experts]
    
    # 获取每个token的最优专家
    _, selected_experts = torch.topk(router_probs, k=2, dim=-1)  # [b*s]
    
    # 创建one-hot矩阵表示选中的专家
    mask = torch.nn.functional.one_hot(selected_experts, num_experts).float()  # [b*s, num_experts]
    
    # 计算每个专家的期望负载 (理想情况下应该是 1/num_experts)
    expected_load = torch.ones_like(router_probs) / num_experts
    
    # 计算实际负载 (每个专家处理的token数量除以总token数量)
    # 在batch维度上计算平均值
    actual_load = mask.mean(dim=0)  # [num_experts]
    
    # 计算auxiliary loss
    # 这会惩罚负载分布与期望负载的差异
    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * num_experts
    
    # 计算z_loss (可选)
    # 这会惩罚过大的路由logits
    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001  # 可调整的超参数
    
    # 总损失
    total_loss = aux_loss + z_loss * z_loss_weight
    
    return total_loss