# -*- coding:utf-8  -*-
"""
Time: 2022/10/18 19:42
Author: Yimohanser
Software: PyCharm
"""
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from lib.optim import Ranger
from lib.datasets.ISIC import ISIC_Dataset
from lib.utils.utils import create_logging, read_logging
from lib.core.function import train, val
from lib.sseg import SSegmentationSet
from lib.criterion import CriterionSet
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    # Model parameters
    parser.add_argument('--model', default='HST_UNet', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--pretrained', default='', type=str)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 0.001)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--root', default=r'D:\迅雷下载\ISIC\train', type=str,
                        help='dataset path')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--workers', default=2, type=int)


    # Cudnn parameters
    parser.add_argument('--BENCHMARK', type=bool, default=True)
    parser.add_argument('--DETERMINISTIC', type=bool, default=False)
    parser.add_argument('--ENABLED', type=bool, default=True)

    # Logging parameters
    parser.add_argument('--log_path', default='./saveModels/logging/',
                        help='path where to tensorboard log')
    parser.add_argument('--log_file', type=str, default='')

    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.resume:
        count = 0
        while os.path.exists(
                args.log_path + args.model + "_ISIC_ver_" + str(count)):
            count += 1
        args.log_path = args.log_path + args.model + "_ISIC_ver_" + str(count)
        os.mkdir(args.log_path)
    print('chkpt path: ', args.log_path)

    writer = SummaryWriter(args.log_path)
    if args.resume:
        logger = read_logging(args.log_file)
    else:
        logger = create_logging(args, args.model, 'train')

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # cudnn related setting
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED

    device = torch.device(args.device)
    logger.info('Use device:{}'.format(args.device))

    # build model
    model = SSegmentationSet(model=args.model.lower(),
                             num_classes=1,
                             pretrained=args.pretrained,
                             img_size=args.img_size)

    # prepare data
    train_dataset = ISIC_Dataset(root=args.root,
                                 mean=[0.709, 0.581, 0.535], std=[0.157, 0.166, 0.181],
                                 img_size=args.img_size, mode='train', fold=args.fold)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=True, pin_memory=True, drop_last=True)

    val_dataset = ISIC_Dataset(root=args.root,
                               mean=[0.709, 0.581, 0.535], std=[0.157, 0.166, 0.181],
                               img_size=args.img_size, mode='val', fold=args.fold)
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True
    )

    # criterion
    criterion = CriterionSet(loss='bce_dice_loss')

    # optimizer
    model = model.to(device)
    optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / (x + 1))

    if args.resume:
        checkpoint = torch.load(args.log_path + '/checkpoint.pth', map_location='cpu')
        epoch_start = checkpoint['epoch']
        max_iou = checkpoint['IoU']
        max_dice = checkpoint['Dice']
        max_f1 = checkpoint['F1']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('------------RESUME TRAINING------------')
    else:
        epoch_start = 0
        max_iou = 0.
        max_dice = 0.
        max_acc = 0.
        max_f1 = 0.

    # amp training
    scaler = GradScaler()
    for epoch in range(epoch_start, args.epochs):
        print('\nEpoch:{}/{}:'.format(epoch, args.epochs))
        train(model=model, optimizer=optimizer, criterion=criterion,
              dataloader=trainloader, epoch=epoch, logging=logger,
              log_path=args.log_path, writer=writer, scaler=scaler,
              device=device)

        scheduler.step(epoch=epoch + 1)

        val_log = val(model=model, dataloader=valloader, criterion=criterion,
                       writer=writer, logging=logger, epoch=epoch, device=device,
                       log_path=args.log_path)

        IoU = val_log['IoU']
        Dice = val_log['Dice']
        F1 = val_log['F1']

        if Dice >= max_dice and F1 >= max_f1 and IoU >= max_iou:
            print('--------------- Save Best! ---------------')
            max_iou = max(max_iou, IoU)
            max_dice = max(max_dice, Dice)
            max_f1 = max(max_f1, F1)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'IoU': max_iou,
                'Dice': max_dice,
                'F1': max_f1
            }, os.path.join(args.log_path, 'Best.pth'))

        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'IoU': max_iou,
            'Dice': max_dice,
            'F1': max_f1
        }, os.path.join(args.log_path, 'checkpoint.pth'))


if __name__ == "__main__":
    main()
