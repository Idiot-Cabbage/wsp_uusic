# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from tkinter import image_names
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation, visual_segmentation_binary, visual_segmentation_sets, visual_segmentation_sets_with_pt,visual_compare
from einops import rearrange
from utils.generate_prompts import get_click_prompt
import time
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def obtain_patien_id(filename):
    if "-" in filename: # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid

def eval_mask_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    hds = np.zeros(opt.classes)
    ious, accs, ses, sps = np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes)
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[1] += iou
            accs[1] += acc
            ses[1] += se
            sps[1] += sp
            hds[1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
        eval_number = eval_number + b
    dices = dices / eval_number
    hds = hds / eval_number
    ious, accs, ses, sps = ious/eval_number, accs/eval_number, ses/eval_number, sps/eval_number
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    mean_hdis = np.mean(hds[1:])
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious[1:]), np.mean(accs[1:]), np.mean(ses[1:]), np.mean(sps[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        return mean_dice, mean_iou, mean_acc, mean_se, mean_sp


def eval_mask_slice2(valloader, model, criterion, opt, args,epoch):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        # class_id = datapack['class_id']
        image_filename = datapack['image_name']
        print('batch_idx [{}/{}]'.format(batch_idx,len(valloader)))
        # pt =  datapack['pt']# get_click_prompt(datapack, opt)
        with torch.no_grad():
            start_time = time.time()
            # pred = model(imgs, pt)
            pred = model(imgs)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            # gt_i[gt[j:j+1, :, :] == 1] = 255
            # 添加调试信息
            # print(f"gt.shape: {gt.shape}, gt_i.shape: {gt_i.shape}")
            # print(f"j: {j}")

            # 安全地进行布尔索引
            # 创建掩码
            mask = gt[j:j+1, :, :] == 1
            # print(f"mask.shape: {mask.shape}")
            
            # 调整掩码大小以匹配 gt_i
            if mask.shape[1:] != gt_i.shape[1:]:
                from scipy.ndimage import zoom
                # 计算缩放比例
                zoom_factors = (1, gt_i.shape[1]/mask.shape[1], gt_i.shape[2]/mask.shape[2])
                print(f"调整掩码尺寸: {mask.shape} -> {gt_i.shape}, 缩放因子: {zoom_factors}")
                # 使用最近邻插值调整掩码大小（保持二值特性）
                mask = zoom(mask.astype(np.uint8), zoom_factors, order=0).astype(bool)
            
            # 应用调整后的掩码
            gt_i[mask] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[eval_number+j, 1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            # # 保存图片
            visual_compare(image_filename[j],pred_i,gt_i,opt,epoch)
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets(seg[j:j+1, :, :], image_filename[j], opt)
        eval_number = eval_number + b
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_F_score(valloader, model, criterion, opt, args):
    model.eval()
    model.eval()
    val_losses = 0
    all_preds = []
    all_labels = []
    eval_number = 0
    sum_time = 0

    # 统计不同指标
    f1_scores = np.zeros(opt.classes)
    accuracy = 0
    precision = 0
    recall = 0
    # 逐batch进行评估
    for batch_idx, datapack in enumerate(valloader):
        imgs = datapack['crop_image'].to(dtype=torch.float32, device=opt.device)
        class_labels = torch.as_tensor(datapack['class_label'], dtype=torch.int64, device=opt.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs)
            sum_time += (time.time() - start_time)

        # 计算损失
        val_loss = criterion(pred, class_labels)
        val_losses += val_loss.item()

        # 获取预测的类别标签
        _, predicted = torch.max(pred, dim=1)  # 选择最大概率对应的类标签

        # 统计预测和真实标签
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(class_labels.cpu().numpy())

        eval_number += imgs.size(0)

    # 合并所有批次的预测结果与标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算分类指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 计算平均损失
    val_losses = val_losses / (batch_idx + 1)

    # 输出评估指标
    print(f"Evaluation speed: {eval_number / sum_time:.2f} samples/sec")
    # print(f"Validation loss: {val_losses:.4f}")
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    # print(f"Precision (macro): {precision * 100:.2f}%")
    # print(f"Recall (macro): {recall * 100:.2f}%")
    # print(f"F1-score (macro): {f1 * 100:.2f}%")

    # 计算每个类的 F1 分数
    for i in range(opt.classes):
        f1_scores[i] = f1_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)


    # 返回相关的指标
    return f1_scores, accuracy, precision, recall, f1, val_losses



def eval_camus_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        # predict = torch.sigmoid(pred['masks'])
        # predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = predict[:, 0, :, :] > 0.5  # (b, h, w)

        predict = F.softmax(pred['masks'], dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum == 3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 5000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt =  datapack['pt']
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        

        # predict = F.softmax(pred['masks'], dim=1)
        # pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j]))
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        pt = get_click_prompt(datapack, opt)
        image_filename = datapack['image_name']

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dices[eval_number+j, 1] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_camus_samed(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    classes = 4
    dices = np.zeros(classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    tns, fns = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    hds = np.zeros((patientnumber, classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        image_filename = datapack['image_name']
        class_id = datapack['class_id']
        
        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt, bbox)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum ==3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
        eval_number = eval_number + b
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    print("test speed", eval_number/sum_time)
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def get_eval(valloader, model, criterion, opt, args,epoch = 0):
    if args.modelname == "SAMed":
        if opt.eval_mode == "camusmulti":
            opt.eval_mode = "camus_samed"
        else:
            opt.eval_mode = "slice"
    if args.modelname == "Resnet18" or args.modelname == "Vit":
        opt.eval_mode = "F-score"
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice2(valloader, model, criterion, opt, args,epoch)
    elif opt.eval_mode == "slice":
        return eval_slice(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camusmulti":
        return eval_camus_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "patient":
        return eval_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camus_samed":
        return eval_camus_samed(valloader, model, criterion, opt, args)
    elif opt.eval_mode =="F-score":
        return eval_F_score(valloader, model, criterion, opt, args)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)