# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Samus, TwoWayTransformer
from torch.nn import functional as F
from .modeling.Auto_Prompt_Generator import AutoPromptGenerator


def build_samus_vit_h(args, checkpoint=None):
    return _build_samus(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_samus = build_samus_vit_h


def build_samus_vit_l(args, checkpoint=None):
    return _build_samus(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_samus_vit_b(args, checkpoint=None):
    return _build_samus(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


samus_model_registry = {
    "default": build_samus_vit_h,
    "vit_h": build_samus_vit_h,
    "vit_l": build_samus_vit_l,
    "vit_b": build_samus_vit_b,
}


def _build_samus(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 224
    # patch_size = image_size//32
    patch_size=16
    image_embedding_size = 14
    samus = Samus(
       image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=224,  # 固定为224
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=16,  # 固定为16
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,  # 224//16 = 14
            out_chans=prompt_embed_dim,
        ),
        auto_prompt_generator= AutoPromptGenerator(
            embed_dim=prompt_embed_dim,
            depth= 2,
            num_heads= 8,
            mlp_ratio= 4,
            batchsize= args.batch_size,
            task_number = 2,
            device= args.device
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )
    samus.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            samus.load_state_dict(state_dict)
        except:
            new_state_dict = load_from2(samus, state_dict, image_size, patch_size)
            samus.load_state_dict(new_state_dict)
    return samus

def load_from(samus, sam_dict, image_size, patch_size):
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    token_size = int(image_size//patch_size)
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict


# def load_from2(samus, sam_dict, image_size, patch_size): # load the positional embedding
#     samus_dict = samus.state_dict()
#     dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
#     token_size = int(image_size//patch_size)
#       # 启用位置编码插值
#     if 'image_encoder.pos_embed' in dict_trained:
#         pos_embed = dict_trained['image_encoder.pos_embed']
#         current_shape = pos_embed.shape
#         expected_shape = (1, token_size, token_size, pos_embed.shape[-1])
        
#         print(f"位置编码形状: 当前 {current_shape} vs 期望 {expected_shape}")
        
#         if current_shape[1:3] != (token_size, token_size):
#             print(f"执行位置编码插值: {current_shape[1:3]} -> {(token_size, token_size)}")
#             pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
#             pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
#             pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
#             dict_trained['image_encoder.pos_embed'] = pos_embed
#             print(f"插值后形状: {pos_embed.shape}")
#         else:
#             print("位置编码形状匹配，无需插值")
#     rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
#     global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
#     for k in global_rel_pos_keys:
#         rel_pos_params = dict_trained[k]
#         h, w = rel_pos_params.shape
#         rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
#         rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
#         dict_trained[k] = rel_pos_params[0, 0, ...]
#     samus_dict.update(dict_trained)
#     return samus_dict
def load_from2(samus, sam_dict, image_size, patch_size): # load the positional embedding
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    token_size = int(image_size//patch_size)
    
    print(f"权重加载: image_size={image_size}, patch_size={patch_size}, token_size={token_size}")
    
    # 获取模型期望的位置编码形状
    model_pos_embed_shape = None
    if 'image_encoder.pos_embed' in samus_dict:
        model_pos_embed_shape = samus_dict['image_encoder.pos_embed'].shape
        print(f"模型期望的位置编码形状: {model_pos_embed_shape}")
    
    # 处理位置编码
    if 'image_encoder.pos_embed' in dict_trained:
        pos_embed = dict_trained['image_encoder.pos_embed']
        current_shape = pos_embed.shape
        
        print(f"权重中的位置编码形状: {current_shape}")
        
        if model_pos_embed_shape and current_shape != model_pos_embed_shape:
            print(f"需要调整位置编码: {current_shape} -> {model_pos_embed_shape}")
            
            # 获取目标形状
            target_h, target_w = model_pos_embed_shape[1], model_pos_embed_shape[2]
            
            if len(current_shape) == 4:
                # 如果权重是 [1, H, W, C] 格式
                if current_shape[1:3] != (target_h, target_w):
                    print(f"执行4D位置编码插值: {current_shape[1:3]} -> {(target_h, target_w)}")
                    pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, C, H, W]
                    pos_embed = F.interpolate(pos_embed, (target_h, target_w), mode='bilinear', align_corners=False)
                    pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, H, W, C]
                    dict_trained['image_encoder.pos_embed'] = pos_embed
                    print(f"插值后形状: {pos_embed.shape}")
                    
            elif len(current_shape) == 3:
                # 如果权重是 [1, N, C] 格式（标准 ViT 格式）
                print("处理标准 ViT 位置编码格式 [1, N, C]")
                
                num_patches = current_shape[1]
                embed_dim = current_shape[2]
                
                # 检测是否有 class token
                if num_patches == 1025:  # 1024 + 1
                    orig_grid_size = 32  # sqrt(1024)
                    has_class_token = True
                elif num_patches == 1024:
                    orig_grid_size = 32
                    has_class_token = False
                elif num_patches == 197:  # 196 + 1
                    orig_grid_size = 14  # sqrt(196)
                    has_class_token = True
                elif num_patches == 196:
                    orig_grid_size = 14
                    has_class_token = False
                else:
                    orig_grid_size = int(num_patches ** 0.5)
                    has_class_token = False
                
                print(f"检测到: 原始网格={orig_grid_size}x{orig_grid_size}, class_token={has_class_token}")
                
                # 分离 class token（如果有）
                if has_class_token:
                    class_token = pos_embed[:, 0:1, :]
                    pos_tokens = pos_embed[:, 1:, :]
                else:
                    class_token = None
                    pos_tokens = pos_embed
                
                # 重塑为 2D 网格
                pos_tokens = pos_tokens.reshape(1, orig_grid_size, orig_grid_size, embed_dim)
                
                # 插值到目标尺寸
                if orig_grid_size != target_h:
                    print(f"插值位置编码: {orig_grid_size}x{orig_grid_size} -> {target_h}x{target_w}")
                    pos_tokens = pos_tokens.permute(0, 3, 1, 2)  # [1, C, H, W]
                    pos_tokens = F.interpolate(pos_tokens, (target_h, target_w), mode='bilinear', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1)  # [1, H, W, C]
                
                # 保存为目标格式
                dict_trained['image_encoder.pos_embed'] = pos_tokens
                print(f"最终位置编码形状: {pos_tokens.shape}")
                
        else:
            print("位置编码形状匹配，无需处理")
    
    # 处理相对位置编码
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    
    samus_dict.update(dict_trained)
    return samus_dict