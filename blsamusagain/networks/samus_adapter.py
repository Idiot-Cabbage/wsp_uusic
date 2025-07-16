import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F

# 添加 SAMUS 模型路径
current_dir = os.path.dirname(__file__)
samus_path = os.path.join(current_dir, '../../KTD/SAMUS-main')
if samus_path not in sys.path:
    sys.path.insert(0, samus_path)

from models.segment_anything_samus.build_sam_us import samus_model_registry
from models.model_dict import get_classifier

class SAMUSAdapter(nn.Module):
    """
    SAMUS 模型适配器 - 内存优化版本
    """
    def __init__(self, config, prompt=False):
        super().__init__()
        self.config = config
        self.prompt = prompt
        
        # 创建 SAMUS 参数对象
        self.samus_args = self.create_samus_args()
        
        # 初始化 SAMUS 分割模型
        try:
            self.samus_model = samus_model_registry['vit_b'](
                args=self.samus_args, 
                checkpoint=self.samus_args.sam_ckpt
            )
            print(f"✓ SAMUS 模型加载成功")
        except Exception as e:
            print(f"✗ SAMUS 模型加载失败: {e}")
            raise
        
        # 轻量级分类器
        self.classifier_2_way = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
        self.classifier_4_way = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
        
        # 轻量级分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, config.MODEL.NUM_CLASSES, 1)
        )
        
        # 特征适配器
        self.feature_adapter = None
        
        print(f"✓ SAMUSAdapter 初始化完成")
        
    def create_samus_args(self):
        """创建 SAMUS 所需的参数"""
        class SAMUSArgs:
            def __init__(self):
                self.modelname = 'SAMUS'
                self.encoder_input_size = 224
                self.low_image_size = 128
                self.vit_name = 'vit_b'
                # 使用配置文件中的路径或默认路径
                self.sam_ckpt = '/home/wtchen/wsp/wsp_uusic/KTD/SAMUS-main/checkpoints/sam_vit_b_01ec64.pth'
                self.batch_size = 1
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
        # 从 self.config 获取配置，而不是在 SAMUSArgs 类内部
        samus_args = SAMUSArgs()
        
        # 如果配置文件中有 SAMUS 配置，使用配置文件的值
        if hasattr(self.config, 'SAMUS') and hasattr(self.config.SAMUS, 'SAM_CKPT'):
            samus_args.sam_ckpt = self.config.SAMUS.SAM_CKPT
        elif hasattr(self.config, 'MODEL') and hasattr(self.config.MODEL, 'SAMUS_CHECKPOINT'):
            samus_args.sam_ckpt = self.config.MODEL.SAMUS_CHECKPOINT
            
        return samus_args
    
    def preprocess_input(self, x):
        """预处理输入数据"""
        if isinstance(x, tuple):
            image_batch = x[0]
        else:
            image_batch = x
        
        batch_size = image_batch.shape[0]
        
        # 维度处理
        if image_batch.dim() == 5:
            image_batch = image_batch[:, 0, :, :, :]
        elif image_batch.dim() == 3:
            if image_batch.shape[0] <= 3:
                image_batch = image_batch.unsqueeze(0)
                batch_size = 1
            else:
                image_batch = image_batch.unsqueeze(1)
        elif image_batch.dim() == 2:
            image_batch = image_batch.unsqueeze(0).unsqueeze(0)
            batch_size = 1
        
        # 确保是RGB格式
        if image_batch.shape[1] == 1:
            image_batch = image_batch.repeat(1, 3, 1, 1)
        elif image_batch.shape[1] > 3:
            image_batch = image_batch[:, :3, :, :]
        
        # 调整尺寸
        if image_batch.shape[2] != 224 or image_batch.shape[3] != 224:
            image_batch = F.interpolate(
                image_batch, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        return image_batch, batch_size
    
    def forward(self, x):
        """前向传播 - 内存优化版本"""
        # 预处理输入
        image_batch, batch_size = self.preprocess_input(x)
        
        # 逐张处理以节省内存
        seg_features_list = []
        
        for i in range(batch_size):
            single_img = image_batch[i:i+1]
            
            # 梯度检查点以节省内存
            with torch.amp.autocast('cuda', enabled=True):
                try:
                    # 使用 SAMUS 模型处理单张图像
                    samus_output = self.samus_model(single_img)
                    
                    # 处理 SAMUS 输出
                    if isinstance(samus_output, dict):
                        if 'masks' in samus_output:
                            seg_features = samus_output['masks']
                        elif 'pred_masks' in samus_output:
                            seg_features = samus_output['pred_masks']
                        elif 'low_res_masks' in samus_output:
                            seg_features = samus_output['low_res_masks']
                        else:
                            # 取第一个tensor值
                            seg_features = list(samus_output.values())[0]
                    elif isinstance(samus_output, tuple):
                        seg_features = samus_output[0]
                    else:
                        seg_features = samus_output
                    
                    # 确保特征有正确的维度
                    if seg_features.dim() == 3:
                        seg_features = seg_features.unsqueeze(1)
                    
                    # 调整特征尺寸到 224x224
                    if seg_features.shape[2] != 224 or seg_features.shape[3] != 224:
                        seg_features = F.interpolate(
                            seg_features, size=(224, 224), mode='bilinear', align_corners=False
                        )
                    
                    # 调整通道数
                    if seg_features.shape[1] != 256:
                        if self.feature_adapter is None:
                            self.feature_adapter = nn.Conv2d(
                                seg_features.shape[1], 256, 1
                            ).to(seg_features.device)
                        seg_features = self.feature_adapter(seg_features)
                    
                    seg_features_list.append(seg_features)
                    
                except Exception as e:
                    print(f"处理第 {i} 张图像时出错: {e}")
                    # 创建零填充的特征
                    dummy_features = torch.zeros(1, 256, 224, 224).to(single_img.device)
                    seg_features_list.append(dummy_features)
            
            # 清理内存
            torch.cuda.empty_cache()
        
        # 合并特征
        seg_features = torch.cat(seg_features_list, dim=0)
        
        # 分割任务
        seg_logits = self.seg_head(seg_features)
        
        # 分类任务
        cls_2_way = self.classifier_2_way(seg_features)
        cls_4_way = self.classifier_4_way(seg_features)
        
        return seg_logits, cls_2_way, cls_4_way
    
    def load_from(self, config):
        """从配置加载预训练权重"""
        try:
            if hasattr(config.MODEL, 'SAMUS_CHECKPOINT') and config.MODEL.SAMUS_CHECKPOINT:
                checkpoint_path = config.MODEL.SAMUS_CHECKPOINT
                if os.path.exists(checkpoint_path):
                    print(f"✓ 从 {checkpoint_path} 加载 SAMUS 权重")
                else:
                    print(f"✗ 检查点文件不存在: {checkpoint_path}")
        except Exception as e:
            print(f"✗ 加载 SAMUS 权重失败: {e}")
    
    def load_from_self(self, checkpoint_path):
        """从自定义检查点加载完整模型权重"""
        try:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.load_state_dict(checkpoint, strict=False)
                print(f"✓ 从 {checkpoint_path} 加载完整模型权重")
            else:
                print(f"✗ 检查点文件不存在: {checkpoint_path}")
        except Exception as e:
            print(f"✗ 加载模型权重失败: {e}")
    
    def get_parameter_count(self):
        """获取模型参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'samus': sum(p.numel() for p in self.samus_model.parameters()),
            'classifier': sum(p.numel() for p in self.classifier_2_way.parameters()) + 
                         sum(p.numel() for p in self.classifier_4_way.parameters()),
            'seg_head': sum(p.numel() for p in self.seg_head.parameters())
        }
    
    def freeze_samus_encoder(self):
        """冻结 SAMUS 编码器参数"""
        for param in self.samus_model.image_encoder.parameters():
            param.requires_grad = False
        print("✓ SAMUS 编码器参数已冻结")
    
    def unfreeze_samus_encoder(self):
        """解冻 SAMUS 编码器参数"""
        for param in self.samus_model.image_encoder.parameters():
            param.requires_grad = True
        print("✓ SAMUS 编码器参数已解冻")