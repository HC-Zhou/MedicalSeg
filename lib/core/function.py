# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 14:04
Author: Yimohanser
Software: PyCharm
"""
import math
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from lib.utils.utils import MetricLogger, writeData, plot_img_gray, SmoothedValue
from lib.utils.evalution import get_score


def train(model: nn.Module, optimizer: optim.Optimizer,
          criterion: nn.Module, dataloader: DataLoader,
          epoch, log_path, logging, writer, scaler, device):
    model.train(True)
    optimizer.zero_grad()
    logging.info('Training: Epoch[{}]'.format(epoch))

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for i_iter, (img, mask, ori) in enumerate(
            metric_logger.log_every(
                dataloader,
                print_freq=(len(dataloader) // 50) if len(dataloader) >= 500 else 10,
                header=header
            )):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(img)
            loss = criterion(out, mask)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        pred_mask = out['out'].sigmoid().squeeze(1).detach().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        mask_list = mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)

        metric = get_score(pred=pred_mask, gt=mask_list)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(loss=loss_value)
        metric_logger.update(SE=metric['SE'])
        metric_logger.update(SP=metric['SP'])
        metric_logger.update(Dice=metric['Dice'])
        metric_logger.update(IoU=metric['IoU'])
        metric_logger.update(F1=metric['F1'])

    loss = metric_logger.meters["loss"].global_avg
    metric = {'SE': metric_logger.meters["SE"].global_avg,
              'SP': metric_logger.meters["SP"].global_avg,
              'Dice': metric_logger.meters["Dice"].global_avg,
              'IoU': metric_logger.meters["IoU"].global_avg,
              'F1': metric_logger.meters["F1"].global_avg}

    plot_img_gray(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='train')
    writeData(writer=writer, logger=logging, loss=loss, metric=metric, mode='train', epoch=epoch)
    torch.cuda.empty_cache()
    return {'Dice': metric_logger.meters["Dice"].global_avg,
            'IoU': metric_logger.meters["IoU"].global_avg,
            'SE': metric_logger.meters["SE"].global_avg,
            'SP': metric_logger.meters["SP"].global_avg,
            'loss': metric_logger.meters["loss"].global_avg,
            'F1': metric_logger.meters["F1"].global_avg}


@torch.no_grad()
def val(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
        writer, logging, epoch, log_path, device):
    model.eval()

    logging.info('Valid Epoch[{}]:'.format(epoch))
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    for i_iter, (img, mask, ori) in enumerate(
            metric_logger.log_every(
                dataloader,
                print_freq=(len(dataloader) // 20) if len(dataloader) >= 200 else 10,
                header=header
            )):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        out = model(img)
        loss = criterion(out, mask)

        loss_value = loss.item()

        pred_mask = out['out'].sigmoid().squeeze(1).detach().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        mask_list = mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)

        metric = get_score(pred=pred_mask, gt=mask_list)
        metric_logger.update(loss=loss_value)
        metric_logger.update(SE=metric['SE'])
        metric_logger.update(SP=metric['SP'])
        metric_logger.update(Dice=metric['Dice'])
        metric_logger.update(IoU=metric['IoU'])
        metric_logger.update(F1=metric['F1'])

    loss = metric_logger.meters["loss"].global_avg
    metric = {'SE': metric_logger.meters["SE"].global_avg,
              'SP': metric_logger.meters["SP"].global_avg,
              'Dice': metric_logger.meters["Dice"].global_avg,
              'IoU': metric_logger.meters["IoU"].global_avg,
              'F1': metric_logger.meters["F1"].global_avg}

    plot_img_gray(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='val')
    writeData(writer=writer, logger=logging, loss=loss, metric=metric, mode='val', epoch=epoch)
    torch.cuda.empty_cache()

    return {'Dice': metric_logger.meters["Dice"].global_avg,
            'IoU': metric_logger.meters["IoU"].global_avg,
            'SE': metric_logger.meters["SE"].global_avg,
            'SP': metric_logger.meters["SP"].global_avg,
            'loss': metric_logger.meters["loss"].global_avg,
            'F1': metric_logger.meters["F1"].global_avg}


@torch.no_grad()
def test(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
        writer, logging, epoch, log_path, device):
    model.eval()

    logging.info('Valid Epoch[{}]:'.format(epoch))
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    for i_iter, (img, mask, ori) in enumerate(
            metric_logger.log_every(
                dataloader,
                print_freq=(len(dataloader) // 20) if len(dataloader) >= 200 else 10,
                header=header
            )):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        out = model(img)
        loss = criterion(out, mask)

        loss_value = loss.item()

        pred_mask = out['out'].sigmoid().squeeze(1).detach().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        mask_list = mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)

        metric = get_score(pred=pred_mask, gt=mask_list)
        metric_logger.update(loss=loss_value)
        metric_logger.update(SE=metric['SE'])
        metric_logger.update(SP=metric['SP'])
        metric_logger.update(Dice=metric['Dice'])
        metric_logger.update(IoU=metric['IoU'])
        metric_logger.update(F1=metric['F1'])

    loss = metric_logger.meters["loss"].global_avg
    metric = {'SE': metric_logger.meters["SE"].global_avg,
              'SP': metric_logger.meters["SP"].global_avg,
              'Dice': metric_logger.meters["Dice"].global_avg,
              'IoU': metric_logger.meters["IoU"].global_avg,
              'F1': metric_logger.meters["F1"].global_avg}

    plot_img_gray(img=ori[0], output=pred_mask[0], target=mask_list[0], Spath=log_path, mode='test')
    writeData(writer=writer, logger=logging, loss=loss, metric=metric, mode='test', epoch=epoch)
    torch.cuda.empty_cache()

    return {'Dice': metric_logger.meters["Dice"].global_avg,
            'IoU': metric_logger.meters["IoU"].global_avg,
            'SE': metric_logger.meters["SE"].global_avg,
            'SP': metric_logger.meters["SP"].global_avg,
            'loss': metric_logger.meters["loss"].global_avg,
            'F1': metric_logger.meters["F1"].global_avg}