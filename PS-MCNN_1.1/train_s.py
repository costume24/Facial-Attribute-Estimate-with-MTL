'''
Training script for CelebA
Copyright (c) Ke Xu, 2020
'''

import argparse
import os
import shutil
import time
import random
import math
import numpy as np

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
import torch.nn.functional as F
import models
from math import cos, pi
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# from prefetch_generator import BackgroundGenerator
from celeb_for_s import CelebA
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter
from lfwa import LFWA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('-d',
                    '--data',
                    default='/root/OneDrive/DataSets/CelebA/Anno',
                    type=str)
parser.add_argument('--set',default='c',type=str)
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch',
                    default=64,
                    type=int,
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch',
                    default=64,
                    type=int,
                    help='test batchsize (default: 200)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=5e-4,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--lr-decay',
                    type=str,
                    default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step',
                    type=int,
                    default=10,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[12, 22],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--turning-point',
                    type=int,
                    default=100,
                    help='epoch number from linear to exponential decay mode')
parser.add_argument('--gamma',
                    type=float,
                    default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--focal', default='yes', type=str)
parser.add_argument('--use1x1', default=True, type=bool)
parser.add_argument('--prob',default=0.5,type=float)
parser.add_argument('--adaloss',default='no', type=str)
parser.add_argument('--prelu',default='no',type=str)
parser.add_argument('--order',default='old',type=str)
# Checkpoints
parser.add_argument('-c',
                    '--checkpoint',
                    default='checkpoints_' +
                    time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
                    type=str,
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--pret',
                    default='',
                    type=str)
parser.add_argument('--pres', default='', type=str)
parser.add_argument('--pre4t', default='', type=str)
# Miscs
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
# Device options
parser.add_argument('--gpu-id',
                    default='0',
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--version',
                    default=1,
                    type=int,
                    help='The version of ps-mcnn architecture')
parser.add_argument('--place',
                    default='my',
                    type=str,
                    help='The place where the programm on')
best_prec1 = 0
best_train_acc = 0
best_b_acc_val = 0
label_list = [
    'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Black_Hair',
    'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Gray_Hair',
    'Narrow_Eyes', 'Receding_Hairline', 'Wearing_Hat', 'Big_Nose',
    'High_Cheekbones', 'Pointy_Nose', 'Rosy_Cheeks', 'Sideburns',
    'Wearing_Earrings', 'Big_Lips', 'Double_Chin', 'Goatee', 'Mustache',
    'Mouth_Slightly_Open', 'No_Beard', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', '5_o_Clock_Shadow', 'Attractive', 'Blurry', 'Chubby',
    'Heavy_Makeup', 'Male', 'Oval_Face', 'Pale_Skin', 'Straight_Hair',
    'Smiling', 'Wavy_Hair', 'Young'
]
attr_order = [
    1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 23, 28, 35, 7, 19, 27, 29, 30, 34, 6, 14,
    16, 22, 21, 24, 36, 37, 38, 0, 2, 10, 13, 18, 20, 25, 26, 32, 31, 33, 39
]


def main():
    global args, best_prec1, best_train_acc, best_b_acc_val
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
    if args.version == 30:
        model = models.s_pretrained.psnet().to(device)
        title = args.set+'-psmcnn-30'

    data_path = ''
    if args.set == 'c':
        if args.place == 'deepai':
            data_path = '/root/OneDrive/DataSets/CelebA_Crop/'
        elif args.place == 'my':
            data_path = '/media/xuke/SoftWare/BaiduNetdiskDownload/CelebA_Crop/'
        elif args.place == 'kb541':
            data_path = '/media/E/xuke/CelebA_Crop/'
        elif args.place == 'phd-1':
            data_path = '/media/kb541/data/xuke/CelebA_Crop/'
        elif args.place == 'mist':
            data_path = '/home/mist/CelebA_Crop/'
    elif args.set == 'l':
        if args.place == 'deepai':
            data_path = '/root/OneDrive/DataSets/LFWA/'
        elif args.place == 'my':
            data_path = '/media/xuke/SoftWare/BaiduNetdiskDownload/LFWA/'
        elif args.place == 'kb541':
            data_path = '/media/E/xuke/LFWA/'
        elif args.place == 'phd-1':
            data_path = '/media/kb541/data/xuke/LFWA/'
        elif args.place == 'mist':
            data_path = '/home/mist/LFWA//'
    # model.apply(weight_init)
    # define loss function (criterion) and optimizer
    if args.focal == 'yes':
        print('=> Focal loss enabled')
        criterion = BCEFocalLoss().cuda()
    else:
        print('=> CrossEntrypy loss enabled')
        criterion = nn.CrossEntropyLoss().cuda()


    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                 args.lr,
                                 weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    args.checkpoint = 'checkpoints_v' + '_'.join([str(args.version), args.set,
                                                 args.focal, args.adaloss,
                                                 str(args.epochs), str(args.train_batch),
                                                 '{:.0e}'.format(args.lr), str(args.lr_decay)])
    if not os.path.isdir(args.checkpoint):
        args.checkpoint = mkdir_p(args.checkpoint, 1)

    # resume work
    if args.pres:
        if os.path.isfile(args.pres):
            print("=> loading checkpoint '{}'".format(args.pres))
            checkpoint = torch.load(args.pres)
            optimizer.load_state_dict(checkpoint['optimizer'])

            save_model = checkpoint['state_dict']

            model_dict = model.state_dict()
            state_dict = {}
            for k, v in save_model.items():
                if k in model_dict.keys() and 'fc' not in k:
                    state_dict[k] = v
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print("=> loaded checkpoint '{}'".format(args.pres))
        else:
            print("=> no checkpoint found at '{}'".format(args.pres))
    if args.pret:
        if os.path.isfile(args.pret):
            print("=> loading checkpoint '{}'".format(args.pret))
            checkpoint = torch.load(args.pret)
            optimizer.load_state_dict(checkpoint['optimizer'])

            save_model = checkpoint['state_dict']

            model_dict = model.state_dict()
            state_dict = {}
            for k, v in save_model.items():
                if k in model_dict.keys() and 'fc' not in k:
                    state_dict[k] = v
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print("=> loaded checkpoint '{}'".format(args.pret))
        else:
            print("=> no checkpoint found at '{}'".format(args.pret))
    if args.pre4t:
        if os.path.isfile(args.pre4t):
            print("=> loading checkpoint '{}'".format(args.pre4t))
            checkpoint = torch.load(args.pre4t)
            optimizer.load_state_dict(checkpoint['optimizer'])

            save_model = checkpoint['state_dict']

            model_dict = model.state_dict()
            state_dict = {}
            for k, v in save_model.items():
                if 't_conv' in k or 's_conv' in k:
                    kk = k.split('.')
                    kk.insert(1,'0')
                    for i in range(4):
                        kk[1]  = str(i)
                        kkk = '.'.join(kk)
                        if kkk in model_dict.keys():
                            state_dict[kkk] = v
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print("=> loaded checkpoint '{}'".format(args.pre4t))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre4t))
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names([
        'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
        'Valid Acc.'
    ])

    cudnn.benchmark = True

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.383, 0.426, 0.506],
                                     std=[0.290, 0.290, 0.311])
    if args.set == 'c':
        train_dataset = CelebA(
            data_path, 'list_attr_celeba_train.txt', 'identity_CelebA_train.txt',
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((160, 192)),
                transforms.ToTensor(),
                normalize,
                RandomErasing(args.prob),
            ]))
        val_dataset = CelebA(
            data_path, 'list_attr_celeba_val.txt', 'identity_CelebA_val.txt',
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomCrop((160, 192)),
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.set == 'l':
        train_dataset = LFWA(
            data_path, 'train.txt',
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((160, 192)),
                transforms.ToTensor(),
                normalize,
                RandomErasing(args.prob),
            ]))
        val_dataset = LFWA(
            data_path,'val.txt',
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



    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))
    count_train = 0
    count_val = 0

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch)


        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)


        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    bar = Bar('Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        # measure accuracy and record loss
        loss = F.nll_loss(output, target)
        prec1 = accuracy(output.data, target, topk = (1, ))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0].data.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss,
                    top1=prec1[0],
                    )
        bar.next()
    bar.finish()
    return (loss, prec1[0])


def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(40)]
    top1 = [AverageMeter() for _ in range(40)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            loss = []
            prec1 = []
            for j in range(len(output)):
                loss.append(criterion(output[j], target[:, j]))
                prec1.append(accuracy(output[j], target[:, j], topk=(1,)))

                losses[j].update(loss[j].item(), input.size(0))
                top1[j].update(prec1[j][0].item(), input.size(0))
            losses_avg = [losses[k].avg for k in range(len(losses))]
            top1_avg = [top1[k].avg for k in range(len(top1))]
            loss_avg = sum(losses_avg) / len(losses_avg)
            prec1_avg = sum(top1_avg) / len(top1_avg)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss_avg,
                    top1=prec1_avg,
                    )
        bar.next()
    bar.finish()
    return (loss_avg, prec1_avg)

def save_checkpoint(state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


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
    elif args.lr_decay=='warmup':
        if epoch<2:
            lr=1e-5
        elif epoch == 2:
            lr=args.lr
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def rank(input, mode):
    input = list(input.cpu().numpy())
    orders = [0] * 6
    w = open(os.path.join(args.checkpoint, mode + '.txt'),'w')
    if mode.startswith('A'):
        if args.set == 'c':
            path = './origin.txt'
        else:
            path = './origin_lfwa.txt'
        with open(path) as f:
            sline = f.readline()
            i = 0
            while sline:
                line = sline.split()
                aine = list(map(float, line[1:]))
                aine.append(100 * (1 - input[i]))
                aine = sorted(aine)
                order = aine.index(100 * (1 - input[i])) + 1
                orders[order - 1] += 1
                w.writelines(sline[:-1] + ' | ' + str(round(100 * (1 - input[i]),2)) + ' | ' +str(round(input[i],4)) + ' | #' + str(order)+'\n')
                sline = f.readline()
                i += 1
            for j in range(6):
                w.writelines('{}st: {}\n'.format(j+1,orders[j]))
    else:
        for i in range(len(input)):
            w.writelines(label_list[i] + ': ' + str(round(100 * (1 - input[i]),4)) +' ' + str(round(input[i],4))+'\n')
    w.close()

def make_confusion_matrix(y_true,y_pred):
    conf_mat_dict={}
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    for label_col in range(len(label_list)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        conf_mat_dict[label_list[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

    with open(os.path.join(args.checkpoint,'confusion.txt'), 'w') as f:
        for label, matrix in conf_mat_dict.items():
            f.writelines("Confusion matrix for label {}:\n".format(label))
            f.writelines(str(matrix)+'\n')
            f.writelines('==============================\n')

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, x, t):
        n,_=x.size()
        p = x.sigmoid()
        onehot = torch.FloatTensor(n, 2).cuda()
        onehot.zero_()
        t = t.view(n,-1)
        onehot.scatter_(1, t, 1)
        t = onehot
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = self.alpha * t + (1 - self.alpha) * (1 - t
                                       )  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(p,
                                                  t,
                                                  w.detach_(),
                                                  reduction=self.reduction)
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.383, 0.426, 0.506]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

if __name__ == '__main__':
    main()
