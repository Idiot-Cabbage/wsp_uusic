import numpy as np
import torch
from medpy import metric
import torch.nn as nn
import cv2


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice, True
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0, False
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0, True
    else:
        return 0, False


def omni_seg_test(image, label, net, classes, ClassStartIndex=1, test_save_path=None, case=None,
                  prompt=False,
                  type_prompt=None,
                  nature_prompt=None,
                  position_prompt=None,
                  task_prompt=None
                  ):
    label = label.squeeze(0).cpu().detach().numpy()
    image_save = image.squeeze(0).cpu().detach().numpy()
    input = image.cuda()
    if prompt:
        position_prompt = position_prompt.cuda()
        task_prompt = task_prompt.cuda()
        type_prompt = type_prompt.cuda()
        nature_prompt = nature_prompt.cuda()
    net.eval()
    with torch.no_grad():
        if prompt:
            seg_out = net((input, position_prompt, task_prompt, type_prompt, nature_prompt))[0]
        else:
            seg_out = net(input)[0]
        out_label_back_transform = torch.cat(
            [seg_out[:, 0:1], seg_out[:, ClassStartIndex:ClassStartIndex+classes-1]], axis=1)
        # out = torch.argmax(torch.softmax(out_label_back_transform, dim=1), dim=1).squeeze(0)
        out = out_label_back_transform[:,0,:,:]>0.5
        prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        image = (image_save - np.min(image_save)) / (np.max(image_save) - np.min(image_save))
        cv2.imwrite(test_save_path + '/'+case + "_pred.png", (prediction*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_img.png", ((image.squeeze(0))*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_gt.png", (label*255).astype(np.uint8))
    return metric_list
import torch.optim as optim
def setup_multi_lr_optimizer(model, base_lr):
    """设置多层级学习率优化器"""
    # 分离参数
    seg_params = []
    cls_params = []
    
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['cls', 'classification', 'head']):
            cls_params.append(param)
        else:
            seg_params.append(param)
    
    # 创建参数组
    param_groups = [
        {
            'params': seg_params, 
            'lr': base_lr, 
            'name': 'segmentation',
            'weight_decay': 1e-4
        },
        {
            'params': cls_params, 
            'lr': base_lr * 0.01,  # 分类学习率大幅降低
            'name': 'classification',
            'weight_decay': 1e-3  # 分类任务增加正则化
        }
    ]
    
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
    return optimizer


# 特定数据集的学习率调整策略
def adjust_lr_by_dataset_before(optimizer, currentnum, base_lr, dataset_info=None):
    """根据数据集调整学习率"""
    # 基础学习率衰减
    base_decay = 1
    if currentnum!=0:
        return
        
    
    # 数据集特定倍数
    dataset_multipliers = {
        "Appendix": 0.1,  # 过拟合严重
        "private_Breast_luminal": 0.1,  # 4分类难度大
        
    }
    
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'segmentation':
            param_group['lr'] = base_lr * base_decay
            lr= param_group['lr']
        elif param_group['name'] == 'classification':
            # 分类任务基础学习率就低，再根据数据集调整
            dataset_mult = 1.0
            if dataset_info and dataset_info in dataset_multipliers:
                dataset_mult = dataset_multipliers[dataset_info]
            param_group['lr'] = base_lr * 0.1 * base_decay * dataset_mult
            lr= param_group['lr']
    
   


# 特定数据集的学习率调整策略
def adjust_lr_by_dataset(optimizer, global_iter_num, max_iterations, base_lr, dataset_info=None):
    """根据数据集调整学习率"""
    # 基础学习率衰减
    base_decay = (1.0 - global_iter_num / max_iterations) ** 0.9
    
    # 数据集特定倍数
    dataset_multipliers = {
        "Appendix": 0.1,  # 过拟合严重
        "private_Breast_luminal": 0.1,  # 4分类难度大
        
    }
    
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'segmentation':
            param_group['lr'] = base_lr * base_decay
            lr= param_group['lr']
        elif param_group['name'] == 'classification':
            # 分类任务基础学习率就低，再根据数据集调整
            dataset_mult = 1.0
            if dataset_info and dataset_info in dataset_multipliers:
                dataset_mult = dataset_multipliers[dataset_info]
            param_group['lr'] = base_lr * 0.1 * base_decay * dataset_mult
            lr= param_group['lr']
    
    return   lr      

