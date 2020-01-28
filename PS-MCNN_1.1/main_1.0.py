'''
Training script for CelebA
Copyright (c) Ke Xu, 2020
'''

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from math import cos, pi

from prefetch_generator import BackgroundGenerator
from celeba import CelebA, TensorSampler, data_prefetcher
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('-d', '--data', default='/media/xuke/Files/Final/DataSet', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=64, type=int, help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', type=str, default='step', help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=10, help='interval for learning rate decay in step mode')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--turning-point',
                    type=int,
                    default=100,
                    help='epoch number from linear to exponential decay mode')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c',
                    '--checkpoint',
                    default='checkpoints_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
                    type=str,
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

best_prec1 = 0

label_list = [
    'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Gray_Hair', 'Narrow_Eyes', 'Receding_Hairline', 'Wearing_Hat', 'Big_Nose', 'High_Cheekbones',
    'Pointy_Nose', 'Rosy_Cheeks', 'Sideburns', 'Wearing_Earrings', 'Big_Lips', 'Double_Chin', 'Goatee', 'Mustache',
    'Mouth_Slightly_Open', 'No_Beard', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', '5_o_Clock_Shadow',
    'Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 'Oval_Face', 'Pale_Skin', 'Straight_Hair', 'Smiling',
    'Wavy_Hair', 'Young'
]


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # create model
    model = models.psmcnn.psnet().to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'CelebA-psmcnn'
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # resume work
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epochs', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True

    # Data loading code
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.383, 0.426, 0.506], std=[0.290, 0.290, 0.311])

    train_dataset = CelebA(
        args.data, 'list_attr_celeba_train.txt', 'identity_CelebA_train.txt', 'train',
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((160, 192)),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = CelebA(
        args.data, 'list_attr_celeba_val.txt', 'identity_CelebA_val.txt', 'valid',
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop((160, 192)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_dataset = CelebA(
        args.data, 'list_attr_celeba_test.txt', 'identity_CelebA_test.txt', 'test',
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop((160, 192)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.test_batch,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              pin_memory=False)

    # if args.evaluate:
    #     validate(test_loader, model, criterion,)
    #     return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, model, criterion, writer)

        # append logger file
        logger.append([epoch, lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': 'psmcnn',
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint=args.checkpoint)
    print('[*] Training task finished at ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    bar = Bar('Training', max=len(train_loader))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    train_total = 0.0
    train_correct = 0.0
    count = 0
    for i, (input, target) in enumerate(train_loader):
        # 预加载代码，暂时不用，会爆显存
        # prefetcher = data_prefetcher(train_loader)
        # input, target = prefetcher.next()
        # i = 0
        # while input is not None:

        # measure data loading time
        optimizer.zero_grad()

        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output_0, output_1, output_2, output_3 = model.forward(input)
        output_0 = output_0.view(-1, 2, 13)
        output_1 = output_1.view(-1, 2, 6)
        output_2 = output_2.view(-1, 2, 9)
        output_3 = output_3.view(-1, 2, 12)
        output = torch.cat([output_0, output_1, output_2, output_3], 2)

        # measure accuracy and record loss
        loss = 0.0
        loss_attr = [0.0 for i in range(40)]
        for k in range(40):
            loss_attr[k] += criterion(output[:, :, k], target[:, k].long())
            loss += loss_attr[k]
        loss = loss.requires_grad_()
        _, pred = torch.max(output, 1)
        correct_single = torch.sum(pred == target, 0, dtype=torch.float32) / output.size(0)
        train_correct += torch.sum(pred == target, dtype=torch.float32).item()  # num_classes need you to define

        train_total += output.size(0)
        cls_train_Accuracy = train_correct / train_total / 40.0
        loss_avg = sum(loss_attr) / len(loss_attr)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        acc_dic = {'train accuracy': cls_train_Accuracy}
        for ii in range(len(correct_single)):
            acc_dic[label_list[ii]] = correct_single[ii]
        writer.add_scalars('loss', {'train loss': loss_avg}, count + 1)
        writer.add_scalars('acc', acc_dic, count + 1)
        count += 1
        # plot progress

        # Best and worst performance
        best_acc_id = torch.argmax(correct_single)
        best_attr = label_list[best_acc_id]
        best_acc = correct_single[best_acc_id]

        worst_acc_id = torch.argmin(correct_single)
        worst_attr = label_list[worst_acc_id]
        worst_acc = correct_single[worst_acc_id]
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | '\
        'top1: {top1: .4f} | Best Attr: {best_attr:<19} | Best Acc:{best_acc:.4f} | Worst Attr: {worst_attr:<19} | Worst Acc: {worst_acc:.4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss_avg,
            top1=cls_train_Accuracy,
            best_attr=best_attr,
            best_acc=best_acc,
            worst_attr=worst_attr,
            worst_acc=worst_acc)
        bar.next()

    # i += 1
    # input, target = prefetcher.next()
    bar.finish()
    return (loss_avg, cls_train_Accuracy)


def validate(val_loader, model, criterion, writer):
    bar = Bar('Validating', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(40)]
    top1 = [AverageMeter() for _ in range(40)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        val_total = 0.0
        val_correct = 0.0
        count = 0
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            # output = model(input)
            output_0, output_1, output_2, output_3 = model.forward(input)
            output_0 = output_0.view(-1, 2, 13)
            output_1 = output_1.view(-1, 2, 6)
            output_2 = output_2.view(-1, 2, 9)
            output_3 = output_3.view(-1, 2, 12)
            output = torch.cat([output_0, output_1, output_2, output_3], 2)

            # measure accuracy and record loss
            loss = 0.0
            loss_attr = [0.0 for i in range(40)]
            for k in range(40):
                loss_attr[k] += criterion(output[:, :, k], target[:, k])
                loss += loss_attr[k]
            loss = loss.requires_grad_()
            _, pred = torch.max(output, 1)
            correct_single = torch.sum(pred == target, 0, dtype=torch.float32) / output.size(0)
            val_correct += torch.sum(pred == target,
                                     dtype=torch.float32).item() / 40.0  # num_classes need you to define

            val_total += output.size(0)
            cls_val_Accuracy = val_correct / val_total
            loss_avg = sum(loss_attr) / len(loss_attr)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            acc_dic = {'validate accuracy': cls_val_Accuracy}
            for ii in range(len(correct_single)):
                acc_dic[label_list[ii]] = correct_single[ii]
            writer.add_scalars('loss', {'validate loss': loss_avg}, count + 1)
            writer.add_scalars('acc', acc_dic, count + 1)
            # plot progress
            best_acc_id = torch.argmax(correct_single)
            best_attr = label_list[best_acc_id]
            best_acc = correct_single[best_acc_id]

            worst_acc_id = torch.argmin(correct_single)
            worst_attr = label_list[worst_acc_id]
            worst_acc = correct_single[worst_acc_id]
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | '\
                'top1: {top1: .4f} | Best Attr: {best_attr:<19} | Best Acc:{best_acc:.4f} | Worst Attr: {worst_attr:<19} | Worst Acc: {worst_acc:4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=loss_avg,
                top1=cls_val_Accuracy,
                best_attr=best_attr,
                best_acc=best_acc,
                worst_attr=worst_attr,
                worst_acc=worst_acc
            )
            bar.next()
    bar.finish()
    return (loss_avg, cls_val_Accuracy)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma**(epoch // args.step))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * epoch / args.epochs)) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - epoch / args.epochs)
    elif args.lr_decay == 'linear2exp':
        if epoch < args.turning_point + 1:
            # learning rate decay as 95% at the turning point (1 / 95% = 1.0526)
            lr = args.lr * (1 - epoch / int(args.turning_point * 1.0526))
        else:
            lr *= args.gamma
    elif args.lr_decay == 'schedule':
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
