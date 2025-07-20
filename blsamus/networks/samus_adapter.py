import torch
import torch.nn as nn
import sys
import os

# 添加 SAMUS 模型路径
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../KTD/SAMUS-main'))
# 添加当前目录到系统路径，确保能找到 models 目录
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from modelsbak.segment_anything_samus.build_sam_us import samus_model_registry
from modelsbak.model_dict import get_classifier

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
        
        # 使用简单的线性分类器代替ResNet
        self.classifier_2_way = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        
        self.classifier_4_way = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )
        
        # 适配 baseline 的输出维度
        self.seg_head = nn.Conv2d(256, config.MODEL.NUM_CLASSES, 1)
        
    def create_samus_args(self):
        """创建 SAMUS 所需的参数"""
        class SAMUSArgs:
            def __init__(self):
                self.modelname = 'SAMUS'
                self.encoder_input_size = 224
                self.low_image_size = 128
                self.vit_name = 'vit_b'
                self.sam_ckpt = '/home/wtchen/wsp/wsp_uusic/KTD/SAMUS-main/checkpoints/sam_vit_b_01ec64.pth'
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
    """
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
            samus_output = self.samus_model(image_batch)
            
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
        
        return seg_logits,None,None
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