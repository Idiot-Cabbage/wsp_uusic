import torch
from functools import partial

from .build_sam_us import _build_samus
from .modeling.moe_samus import MoESamus


def build_moesamus_vit_h(args, checkpoint=None):
    return _build_moesamus(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_moesamus = build_moesamus_vit_h


def build_moesamus_vit_l(args, checkpoint=None):
    return _build_moesamus(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_moesamus_vit_b(args, checkpoint=None):
    return _build_moesamus(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


moesamus_model_registry = {
    "default": build_moesamus_vit_h,
    "vit_h": build_moesamus_vit_h,
    "vit_l": build_moesamus_vit_l,
    "vit_b": build_moesamus_vit_b,
}


def _build_moesamus(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    base = _build_samus(
        args,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=checkpoint,
    )
    moe_model = MoESamus(
        base,
        moe_expert_num=getattr(args, "moe_expert_num", 4),
        top_k=getattr(args, "moe_top_k", 1),
        router_temp=getattr(args, "router_temp", 1.0),
    )
    return moe_model


