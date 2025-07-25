from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
# from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import math
from torchvision import transforms
from utils.dataset import RandomGenerator, CenterCropGenerator  


print("当前工作目录:", os.getcwd())
print("脚本所在目录:", os.path.dirname(os.path.abspath(__file__)))
def visualize_dataset_samples(dataloader, dataset_name, num_samples=8):
    """可视化数据集中的前几个样本"""
    import matplotlib.pyplot as plt
    import os
    
    # 创建保存目录
    save_dir = f'dataset_visualization/{dataset_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"正在可视化{dataset_name}数据...")
    
    # 获取一个batch的数据
    dataiter = iter(dataloader)
    datapack = next(dataiter)
    
    # 打印datapack的键以了解其结构
    print(f"Datapack keys: {list(datapack.keys())}")
    
    images = datapack['image'].cpu().numpy()
    masks = datapack['label'].cpu().numpy()
    filenames = datapack['image_name']
    
    # 仅展示指定数量的样本
    for i in range(min(num_samples, len(images))):
        # 获取单个样本
        img = images[i, 0]  # 取第一个通道，假设是灰度图
        mask = masks[i, 0]  # 取第一个通道
        filename = filenames[i]
        
        # 打印统计信息
        print(f"\n样本 {i+1}: {filename}")
        print(f"图像形状: {img.shape}, 值范围: [{img.min():.3f}, {img.max():.3f}]")
        print(f"掩码形状: {mask.shape}, 值范围: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # 统计掩码中1的比例
        mask_ones_ratio = np.sum(mask == 1.0) / mask.size
        print(f"掩码中值为1的像素比例: {mask_ones_ratio:.2%}")
        
        if mask_ones_ratio > 0.99:
            print("警告: 掩码几乎全为1，可能存在问题!")
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示原始图像
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 显示掩码
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'掩码 (min={mask.min():.3f}, max={mask.max():.3f})')
        axes[1].axis('off')
        
        # 显示叠加效果
        axes[2].imshow(img, cmap='gray')
        # mask_overlay = np.zeros((*mask.shape, 4))
        # mask_overlay[mask > 0.5, :] = [1, 0, 0, 0.5]  # 红色半透明
        # axes[2].imshow(mask_overlay)
        axes[2].set_title('叠加效果')
        axes[2].axis('off')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{filename}_sample_{i}.png')
        plt.close()
def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=224, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='UUSIC', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='KTD/SAMUS-main/checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()
    opt = get_config(args.task)
    args.batch_size = opt.batch_size
    args.device = opt.device
    args.base_lr = opt.learning_rate

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = opt.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    # tf_train = transforms.Compose([RandomGenerator(output_size=[args.encoder_input_size, args.encoder_input_size])])
    # tf_val = CenterCropGenerator(output_size=[args.encoder_input_size, args.encoder_input_size])
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    
    
    # # 可视化训练集和验证集样本
    # visualize_dataset_samples(trainloader, 'train')
    # visualize_dataset_samples(valloader, 'val')

    # print("\n数据可视化完成，请检查 'dataset_visualization' 目录")
    # import sys
    # sys.exit(0)  # 可视化后退出程序
    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    # 添加以下代码:
    
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    # 跳过训练直接评估
    skip_training = False  # 临时设置为 True 跳过训练
    if skip_training:
        print("=== 跳过训练，直接进行评估 ===")
        model.eval()
        with torch.no_grad():
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args, epoch=0)
        print('评估结果: val loss:{:.4f}, val dice:{:.4f}'.format(val_losses, mean_dice))
        import sys
        sys.exit(0)  # 评估后直接退出

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            
            # class_labels = torch.as_tensor(datapack['class_label'],dtype = torch.float32, device=opt.device)
            # bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            # pt = get_click_prompt(datapack, opt)
            # print(imgs.shape) # 8 1 256 256
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs) # pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks)
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            # print(batch_idx)
            # for name,param in model.auto_prompt_generator.named_parameters():
            #     print(name, param.requires_grad)  # 输出每个参数是否需要梯度
            #     if 'task_tokens' in name:
            #         print(f"Found {name}, Gradient Sum: {param.grad.abs().sum() if param.grad is not None else 'None'}")
            optimizer.step()
            train_losses += train_loss.item()
            print('batch_idx [{}/{}/{}], train loss:{:.4f}'.format(batch_idx,len(trainloader),epoch,train_loss))
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args,epoch=epoch)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

if __name__ == '__main__':
    main()