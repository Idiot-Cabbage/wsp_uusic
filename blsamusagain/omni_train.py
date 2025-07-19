import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import sys
import torch.nn.functional as F

print("开始导入模块...")

# 添加 SAMUS 模型路径
current_dir = os.path.dirname(__file__)
samus_path = os.path.join(current_dir, '../KTD/SAMUS-main')
if samus_path not in sys.path:
    sys.path.insert(0, samus_path)

print(f"SAMUS 路径: {samus_path}")
print(f"路径是否存在: {os.path.exists(samus_path)}")

try:
    from models.segment_anything_samus.build_sam_us import samus_model_registry
    from models.model_dict import get_classifier
    print("✓ SAMUS 模块导入成功")
except Exception as e:
    print(f"✗ SAMUS 模块导入失败: {e}")
    # 如果 SAMUS 导入失败，使用原始模型
    USE_SAMUS = False
else:
    USE_SAMUS = True

try:
    from networks.omni_vision_transformer import OmniVisionTransformer as ViT_omni
    print("✓ ViT_omni 导入成功")
except Exception as e:
    print(f"✗ ViT_omni 导入失败: {e}")

try:
    from omni_trainer import omni_train
    print("✓ omni_trainer 导入成功")
except Exception as e:
    print(f"✗ omni_trainer 导入失败: {e}")

try:
    from config import get_config
    print("✓ config 导入成功")
except Exception as e:
    print(f"✗ config 导入失败: {e}")

class SAMUSAdapter(nn.Module):
    """
    SAMUS 模型适配器 - 内存优化版本
    """
    def __init__(self, config, prompt=False):
        super().__init__()
        print("初始化 SAMUSAdapter...")
        
        self.config = config
        self.prompt = prompt
        
        # 创建 SAMUS 参数对象
        print("创建 SAMUS 参数...")
        self.samus_args = self.create_samus_args()
        
        # 初始化 SAMUS 分割模型
        print("初始化 SAMUS 模型...")
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
        print("创建分类器...")
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
        print("创建分割头...")
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
                self.sam_ckpt = '/root/autodl-tmp/wsp_uusic/KTD/SAMUS-main/checkpoints/sam_vit_b_01ec64.pth'
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
            
            # 处理单张图像
            try:
                # 使用 SAMUS 模型处理单张图像
                with torch.amp.autocast('cuda', enabled=False):  # 暂时禁用混合精度
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
                
                # 确保数据类型为 float32
                seg_features = seg_features.float()
                
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
                dummy_features = torch.zeros(1, 256, 224, 224, dtype=torch.float32).to(single_img.device)
                seg_features_list.append(dummy_features)
            
            # 清理内存
            torch.cuda.empty_cache()
        
        # 合并特征
        seg_features = torch.cat(seg_features_list, dim=0)
        
        # 确保特征数据类型为 float32
        seg_features = seg_features.float()
        
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

print("开始解析命令行参数...")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/', help='root dir for data')
parser.add_argument('--output_dir', type=str, default='exp_out/samus_default', help='output dir')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/samus_config.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into non-overlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument('--pretrain_ckpt', type=str, help='pretrained checkpoint')

parser.add_argument('--prompt', action='store_true', help='using prompt for training')
parser.add_argument('--adapter_ft', action='store_true', help='using adapter for fine-tuning')

# 添加 SAMUS 特有的参数
parser.add_argument('--use_samus', action='store_true', default=USE_SAMUS, 
                    help='use SAMUS model instead of ViT_omni')
parser.add_argument('--freeze_samus_encoder', action='store_true', 
                    help='freeze SAMUS encoder during training')

# 添加缺少的参数
parser.add_argument('--data_path', type=str, default='', help='path to dataset')
parser.add_argument('--output', type=str, default='', help='output folder')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')

args = parser.parse_args()
print(f"命令行参数解析完成: {args}")

# 确保 output_dir 不为 None
if args.output_dir is None:
    args.output_dir = 'exp_out/samus_default'

# 设置缺失的参数默认值
if not hasattr(args, 'output') or not args.output:
    args.output = args.output_dir

# 尝试加载配置文件
print(f"尝试加载配置文件: {args.cfg}")
try:
    config = get_config(args)
    print("✓ 配置文件加载成功")
except Exception as e:
    print(f"✗ 配置文件加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

if __name__ == "__main__":
    print("进入主程序...")
    
    # 设置GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f"使用 GPU: {args.gpu}")
        
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"GPU 可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU 不可用，将使用 CPU")
    
    # 设置内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("设置随机种子...")
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    # 确保输出目录存在
    print(f"创建输出目录: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # 根据设置选择模型
    if args.use_samus and USE_SAMUS:
        print("=" * 50)
        print("使用 SAMUS 模型")
        print("=" * 50)
        
        # 创建 SAMUS 模型
        print("创建 SAMUS 模型...")
        try:
            net = SAMUSAdapter(
                config,
                prompt=args.prompt,
            ).cuda()
            print("✓ SAMUS 模型创建成功")
        except Exception as e:
            print(f"✗ SAMUS 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        
        # 加载预训练权重
        print("加载预训练权重...")
        if args.pretrain_ckpt is not None:
            net.load_from_self(args.pretrain_ckpt)
        else:
            net.load_from(config)
        
        # 可选：冻结 SAMUS 编码器
        if args.freeze_samus_encoder:
            net.freeze_samus_encoder()
            print("✓ SAMUS 编码器已冻结")
        
        # 打印模型参数信息
        try:
            param_info = net.get_parameter_count()
            print(f"模型参数统计:")
            print(f"  - 总参数: {param_info['total']:,}")
            print(f"  - 可训练参数: {param_info['trainable']:,}")
            print(f"  - SAMUS 参数: {param_info['samus']:,}")
            print(f"  - 分类器参数: {param_info['classifier']:,}")
            print(f"  - 分割头参数: {param_info['seg_head']:,}")
        except Exception as e:
            print(f"无法获取参数统计: {e}")
        
        # 处理适配器微调
        if args.prompt and args.adapter_ft:
            print("启用适配器微调模式...")
            for name, param in net.named_parameters():
                if 'prompt' in name or 'classifier' in name or 'seg_head' in name:
                    param.requires_grad = True
                    print(f"  - 可训练: {name}")
                else:
                    param.requires_grad = False
            
            # 统计实际可训练参数
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"  - 适配器微调可训练参数: {trainable_params:,}")
            
    else:
        print("=" * 50)
        print("使用原始 ViT_omni 模型")
        print("=" * 50)
        
        # 创建原始 ViT_omni 模型
        try:
            net = ViT_omni(
                config,
                prompt=args.prompt,
            ).cuda()
            print("✓ ViT_omni 模型创建成功")
        except Exception as e:
            print(f"✗ ViT_omni 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        
        # 加载预训练权重
        if args.pretrain_ckpt is not None:
            net.load_from_self(args.pretrain_ckpt)
        else:
            net.load_from(config)

        # 处理适配器微调
        if args.prompt and args.adapter_ft:
            for name, param in net.named_parameters():
                if 'prompt' in name:
                    param.requires_grad = True
                    print(name)
                else:
                    param.requires_grad = False

    # 开始训练
    print("=" * 50)
    print(f"开始训练，输出目录: {args.output_dir}")
    print("=" * 50)
    
    try:
        omni_train(args, net, args.output_dir)
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()