import torch
import torch.nn as nn
import sys
import os
from torch.functional import F
import torch.utils.checkpoint as checkpoint
# 添加 SAMUS 模型路径
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../KTD/SAMUS-main'))
# 添加当前目录到系统路径，确保能找到 models 目录
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.segment_anything_samus.build_sam_us import samus_model_registry
from models.model_dict import get_classifier
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, res_scale=None, use_checkpoint=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if res_scale is not None:
            self.res_scale = res_scale(input_resolution, dim)
        else:
            self.res_scale = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.res_scale is not None:
            x = self.res_scale(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class ChannelHalf(nn.Module):
    def __init__(self, input_resolution=None, dim=0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)
        self.input_resolution = input_resolution

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SAMUSAdapter(nn.Module):
    """
    SAMUS 模型适配器，用于适配 baseline 的接口
    """
    def __init__(self, config, prompt=False):
        super().__init__()
        self.config = config
        self.prompt = prompt
        
        # 创建 SAMUS 参数对象
        self.samus_args = self.create_samus_args()
        
        # 初始化 SAMUS 分割模型
        self.samus_model = samus_model_registry['vit_b'](
            args=self.samus_args, 
            checkpoint=self.samus_args.sam_ckpt
        )
        ############################################################################################################
        img_size=224
        patch_size=16
        in_chans=3
        embed_dim=96
        encoder_depths=[2, 2, 2, 2]
        decoder_depths=[2, 2, 2, 2]
        num_heads=[2, 4, 8, 16]
        window_size=8
        self.mlp_ratio=4.
        qkv_bias=True 
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.1
        norm_layer=nn.LayerNorm
        self.patch_norm=True
        ape=False
        use_checkpoint=False
        embed_dim = 32
        self.layers_task_cls_up = nn.ModuleList()

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        dec_dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(decoder_depths))]

        self.num_layers = len(encoder_depths)



        for i_layer in range(self.num_layers):
            if i_layer == 0:
                pass
            else:
                self.layers_task_cls_up.append(
                    BasicLayer(dim=int(embed_dim * 2 ** (self.num_layers-i_layer)),
                               input_resolution=(32,32),
                               depth=decoder_depths[(self.num_layers-i_layer)],
                               num_heads=num_heads[(self.num_layers-i_layer)],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dec_dpr[sum(decoder_depths[:(self.num_layers-i_layer)]):sum(decoder_depths[:(self.num_layers-i_layer) + 1])],
                               norm_layer=norm_layer,
                               res_scale=ChannelHalf if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint
                               ))


        if self.prompt:
            self.dec_prompt_mlp_cls2 = nn.Linear(8+2+2+3, embed_dim*4)
            self.dec_prompt_mlp_seg2_cls3 = nn.Linear(8+2+2+3, embed_dim*2)
        self.norm_task_cls = nn.LayerNorm(embed_dim*2)
        # 使用简单的线性分类器代替ResNet
        self.layers_task_cls_head_2cls = nn.ModuleList([
            nn.Linear(embed_dim*2, 2)
        ])
        self.layers_task_cls_head_4cls = nn.ModuleList([
            nn.Linear(embed_dim*2, 4)
        ])
    
        ##############################################################################################
    def create_samus_args(self):
        """创建 SAMUS 所需的参数"""
        class SAMUSArgs:
            def __init__(self):
                self.modelname = 'SAMUS'
                self.encoder_input_size = 224
                self.low_image_size = 128
                self.vit_name = 'vit_b'
<<<<<<< HEAD
                self.sam_ckpt = '/root/autodl-tmp/wsp_uusic/KTD/SAMUS-main/checkpoints/sam_vit_b_01ec64.pth'
=======
                self.sam_ckpt = '/home/wtchen/wsp/wsp_uusic/KTD/SAMUS-main/checkpoints/sam_vit_b_01ec64.pth'
>>>>>>> 8f14ae534c91fbc05719c7ba34bcb7cc1b607768
                self.batch_size = 1
                self.device = 'cuda'
                
        return SAMUSArgs()
    
    def create_classifier_args(self, num_classes):
        """创建分类器参数"""
        class ClassifierArgs:
            def __init__(self, num_classes):
                self.classifier_name = 'Resnet18'
                self.classifier_size = 256
                self.classifier_classes = num_classes
                
        return ClassifierArgs(num_classes)
    
    # def forward(self, x, prompt_info=None):
        """
        前向传播，适配 baseline 的接口
        """
        batch_size = x.shape[0]
        
        # 分割任务
        seg_outputs = []
        for i in range(batch_size):
            # SAMUS 需要单张图像处理
            single_img = x[i:i+1]
            
            # 使用 SAMUS 进行分割
            with torch.no_grad():
                # 这里需要根据 SAMUS 的具体接口调整
                masks, _ = self.samus_model(single_img, multimask_output=False)
                seg_outputs.append(masks)
        
        seg_logits = torch.cat(seg_outputs, dim=0)
        seg_logits = self.seg_head(seg_logits)
        
        # 分类任务 - 使用全局平均池化的特征
        features = torch.nn.functional.adaptive_avg_pool2d(seg_logits, (1, 1))
        features = features.view(batch_size, -1)
        
        # 通过分类器
        cls_2_way = self.classifier_2_way(features)
        cls_4_way = self.classifier_4_way(features)
        
        return seg_logits, cls_2_way, cls_4_way
    
    # def forward(self, x):
    #     """
    #     前向传播，适配 baseline 的接口
    #     """
    #     # 处理输入，如果是tuple则只取图像部分
    #     if isinstance(x, tuple):
    #         image_batch = x[0]  # 第一个元素是图像
    #         #printf"从tuple中提取图像，原始tuple长度: {len(x)}")
    #     else:
    #         image_batch = x
        
    #     # 详细的调试信息
    #     #printf"输入图像维度: {image_batch.shape}")
    #     #printf"输入图像类型: {type(image_batch)}")
    #     #printf"输入图像数据类型: {image_batch.dtype}")
        
    #     batch_size = image_batch.shape[0]
        
    #     # 处理各种可能的输入维度
    #     if image_batch.dim() == 2:
    #         # 如果是2D (height, width)，添加batch和通道维度
    #         image_batch = image_batch.unsqueeze(0).unsqueeze(0)
    #         #printf"从2D转换为4D后: {image_batch.shape}")
    #         batch_size = 1
            
    #     elif image_batch.dim() == 3:
    #         # 如果是3D，可能是 (batch, height, width) 或 (channels, height, width)
    #         if image_batch.shape[0] <= 3:  # 假设是 (channels, height, width)
    #             image_batch = image_batch.unsqueeze(0)  # 添加batch维度
    #             #printf"从3D (C,H,W) 转换为4D后: {image_batch.shape}")
    #             batch_size = 1
    #         else:  # 假设是 (batch, height, width)
    #             image_batch = image_batch.unsqueeze(1)  # 添加通道维度
    #             #printf"从3D (B,H,W) 转换为4D后: {image_batch.shape}")
                
    #     # elif image_batch.dim() == 4:
    #     #     # 已经是4D，直接使用
    #     #     #printf"输入已经是4D: {image_batch.shape}")
            
    #     elif image_batch.dim() == 5:
    #         # 如果是5D，可能是 (batch, depth, channels, height, width)
    #         # 取中间的切片或第一个切片
    #         image_batch = image_batch[:, 0, :, :, :]  # 取第一个深度切片
    #         #printf"从5D转换为4D后: {image_batch.shape}")
            
    #     else:
    #         raise ValueError(f"无法处理{image_batch.dim()}维输入: {image_batch.shape}")
        
    #     # 确保图像是4维的 (batch, channels, height, width)
    #     if image_batch.dim() != 4:
    #         raise ValueError(f"处理后仍然不是4维: {image_batch.shape}")
        
    #     # 如果是单通道图像，转换为三通道（SAMUS需要RGB输入）
    #     if image_batch.shape[1] == 1:
    #         image_batch = image_batch.repeat(1, 3, 1, 1)
    #         #printf"转换为三通道后: {image_batch.shape}")
    #     elif image_batch.shape[1] > 3:
    #         # 如果通道数超过3，只取前3个通道
    #         image_batch = image_batch[:, :3, :, :]
    #         #printf"截取前3个通道后: {image_batch.shape}")
        
    #     # 确保输入尺寸符合SAMUS要求（通常是224x224）
    #     if image_batch.shape[2] != 224 or image_batch.shape[3] != 224:
    #         image_batch = torch.nn.functional.interpolate(
    #             image_batch, 
    #             size=(224, 224), 
    #             mode='bilinear', 
    #             align_corners=False
    #         )
    #         #printf"调整尺寸后: {image_batch.shape}")
        
    #     # 使用SAMUS模型进行分割
    #     # 使用SAMUS模型进行分割
    #     try:
    #         samus_output = self.samus_model(image_batch)
    #         #printf"SAMUS输出类型: {type(samus_output)}")
            
    #         if isinstance(samus_output, dict):
    #             #printf"SAMUS输出字典键: {list(samus_output.keys())}")
    #             # for key, value in samus_output.items():
    #             #     if hasattr(value, 'shape'):
    #             #         #printf"字典键 '{key}' 的形状: {value.shape}")
    #             #     else:
    #             #         #printf"字典键 '{key}' 的类型: {type(value)}")
                        
    #             # 处理字典输出
    #             possible_keys = ['masks', 'pred_masks', 'logits', 'features', 'low_res_masks', 'output']
    #             seg_features = None
                
    #             for key in possible_keys:
    #                 if key in samus_output:
    #                     seg_features = samus_output[key]
    #                     #printf"使用字典键 '{key}' 作为分割特征")
    #                     break
                
    #             if seg_features is None:
    #                 # 如果没有找到常见键，使用第一个tensor值
    #                 for key, value in samus_output.items():
    #                     if hasattr(value, 'shape') and len(value.shape) >= 3:
    #                         seg_features = value
    #                         #printf"使用字典键 '{key}' 作为分割特征（自动选择）")
    #                         break
                
    #             if seg_features is None:
    #                 raise ValueError(f"无法从SAMUS输出字典中找到合适的分割特征。可用键: {list(samus_output.keys())}")
                    
    #         elif isinstance(samus_output, tuple):
    #             #printf"SAMUS输出tuple长度: {len(samus_output)}")
    #             # for i, output in enumerate(samus_output):
    #             #     if hasattr(output, 'shape'):
    #             #         #printf"输出{i}形状: {output.shape}")
    #             #     else:
    #             #         #printf"输出{i}类型: {type(output)}")
    #             seg_features = samus_output[0]
    #         else:
    #             # if hasattr(samus_output, 'shape'):
    #             #     #printf"SAMUS输出形状: {samus_output.shape}")
    #             # else:
    #             #     #printf"SAMUS输出类型: {type(samus_output)}")
    #             seg_features = samus_output
                
    #     except Exception as e:
    #         print(f"SAMUS模型调用错误: {e}")
    #         print(f"输入形状: {image_batch.shape}")
    #         raise

    #     #printf"处理后的分割特征形状: {seg_features.shape}")
        
      
    #     if seg_features.dim() == 3:
    #         seg_features = seg_features.unsqueeze(1)
    #         #printf"添加通道维度后的特征形状: {seg_features.shape}")
        
    #     # 调整通道数以匹配seg_head的输入
    #     if seg_features.shape[1] != 256:
    #         if not hasattr(self, 'feature_adapter'):
    #             self.feature_adapter = nn.Conv2d(seg_features.shape[1], 256, 1).to(seg_features.device)
    #         seg_features = self.feature_adapter(seg_features)
    #         #printf"适配后的特征形状: {seg_features.shape}")
        
    #     # 通过分割头生成最终分割输出
    #     seg_logits = self.seg_head(seg_features)
    #     #printf"分割输出形状: {seg_logits.shape}")
        
    #     # 分类任务
    #     pooled_features = torch.nn.functional.adaptive_avg_pool2d(seg_features, (1, 1))
    #     pooled_features = pooled_features.view(batch_size, -1)
    #     #printf"池化特征形状: {pooled_features.shape}")

    #     cls_2_way = self.classifier_2_way(pooled_features)
    #     cls_4_way = self.classifier_4_way(pooled_features)
        
    #     #printf"2分类输出形状: {cls_2_way.shape}")
    #     #printf"4分类输出形状: {cls_4_way.shape}")
        
    #     return seg_logits, cls_2_way, cls_4_way
    
    def forward(self, x):
        """
        前向传播，支持分布式训练
        """
        # 处理输入
        if isinstance(x, tuple):
            image_batch = x[0]
        else:
            image_batch = x
        
        batch_size = image_batch.shape[0]
        
        # 维度处理（保持原有逻辑）
        if image_batch.dim() == 5:
            image_batch = image_batch.squeeze(1)
        elif image_batch.dim() == 3:
            if image_batch.shape[0] <= 3:
                image_batch = image_batch.unsqueeze(0)
                batch_size = 1
            else:
                image_batch = image_batch.unsqueeze(1)
        
        # 确保是RGB格式
        if image_batch.shape[1] == 1:
            image_batch = image_batch.repeat(1, 3, 1, 1)
        elif image_batch.shape[1] > 3 and image_batch.shape[-1]==3:
            image_batch = image_batch.permute(0,3,1,2)
        
        # 调整尺寸
        if image_batch.shape[2] != 224 or image_batch.shape[3] != 224:
            image_batch = torch.nn.functional.interpolate(
                image_batch, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # 在分布式环境中，每个GPU处理更小的batch
        # 使用混合精度训练
        # with torch.cuda.amp.autocast():
        try:
            samus_output,image_features = self.samus_model(image_batch) # image_features[batchsize,length,256]
            
            if isinstance(samus_output, dict):
                seg_features = samus_output.get('masks', list(samus_output.values())[0])
            elif isinstance(samus_output, tuple):
                seg_features = samus_output[0]
            else:
                seg_features = samus_output
                
        except torch.cuda.OutOfMemoryError:
            # 如果仍然内存不足，降级到逐张处理
            print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: 内存不足，降级到逐张处理")
            seg_features_list = []
            for i in range(batch_size):
                single_img = image_batch[i:i+1]
                with torch.cuda.amp.autocast():
                    single_output = self.samus_model(single_img)
                    if isinstance(single_output, dict):
                        single_features = single_output.get('masks', list(single_output.values())[0])
                    elif isinstance(single_output, tuple):
                        single_features = single_output[0]
                    else:
                        single_features = single_output
                    seg_features_list.append(single_features)
                torch.cuda.empty_cache()
            seg_features = torch.cat(seg_features_list, dim=0)
        
        # 继续处理...
        if seg_features.dim() == 3:
            seg_features = seg_features.unsqueeze(1)
        
        # if seg_features.shape[1] != 256:
        #     if not hasattr(self, 'feature_adapter'):
        #         self.feature_adapter = nn.Conv2d(seg_features.shape[1], 256, 1).to(seg_features.device)
        #     seg_features = self.feature_adapter(seg_features)
        
        # seg_logits = self.seg_head(seg_features)
        prob = torch.sigmoid(seg_features)
        prob = torch.clamp(prob, 1e-7, 1-1e-7)
        seg_logits = torch.log(torch.cat([1 - prob, prob], dim=1))
        # 分类任务
        # pooled_features = torch.nn.functional.adaptive_avg_pool2d(seg_features, (1, 1))
        # pooled_features = pooled_features.view(batch_size, -1)
        
        # cls_2_way = self.classifier_2_way(pooled_features)
        # cls_4_way = self.classifier_4_way(pooled_features)


        # cls
        x = image_features
        if self.prompt:
            x, position_prompt, task_prompt, type_prompt, nature_prompt = x
        
        for inx, layer_head in enumerate(self.layers_task_cls_up):
            if inx == 0:
                x_cls = layer_head(x)
            else:
                if self.prompt:
                    if inx == 1:
                        x_cls = layer_head(x_cls +
                                           self.dec_prompt_mlp_cls2(torch.cat([position_prompt, task_prompt, type_prompt, nature_prompt], dim=1)).unsqueeze(1))
                    if inx == 2:
                        x_cls = layer_head(x_cls +
                                           self.dec_prompt_mlp_seg2_cls3(torch.cat([position_prompt, task_prompt, type_prompt, nature_prompt], dim=1)).unsqueeze(1))
                else:
                    x_cls = layer_head(x_cls)

        x_cls = self.norm_task_cls(x_cls)

        B, _, _ = x_cls.shape
        x_cls = x_cls.transpose(1, 2)
        x_cls = F.adaptive_avg_pool1d(x_cls, 1).view(B, -1)
        
        x_cls_2_way = self.layers_task_cls_head_2cls[0](x_cls)
        x_cls_4_way = self.layers_task_cls_head_4cls[0](x_cls)


        
        return seg_logits,x_cls_2_way,x_cls_4_way
    def load_from(self, config):
        """加载预训练权重"""
        # 加载 SAMUS 权重
        if hasattr(config, 'SAMUS_CHECKPOINT'):
            checkpoint = torch.load(config.SAMUS_CHECKPOINT)
            self.samus_model.load_state_dict(checkpoint, strict=False)
            
    def load_from_self(self, checkpoint_path):
        """从自定义检查点加载"""
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint, strict=False)