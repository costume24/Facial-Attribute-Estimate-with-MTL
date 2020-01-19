import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np
import os
import time
import math

#import sklearn
from sklearn.metrics import confusion_matrix
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# stage = 1
batch_size = 64

test_batch_size = batch_size
lr = 0.001
weight_decay = 5e-4
best_c_acc = 0.9
best_l_acc = 0.60
stage = 1
epoc = 20
step_size = 8000
step_gamma = 0.1
change_st = 20
dataset = 'c'
v_dataset = dataset
t_dataset = dataset
dev = "computer"

if (dev == 'server'):

    img_root = '/media/E/syliu/celeba_crop/'
    # val_img_root = img_root
    val_l_img_root = '/media/E/syliu/lfw-deepfunneled/'
    val_c_img_root = '/media/E/syliu/celeba_crop/'
    # img_root = '/media/E/syliu/lfw-deepfunneled/'
    train_txt = "train_crop_clean.txt"
    # train_txt = "lfwa/train_crop.txt"
    # val_txt = "val_crop.txt"
    val_l_txt = "lfwa/test_crop.txt"
    val_c_txt = "val_crop.txt"
    # test_txt = "test_crop.txt"
    test_txt = "lfwa/test_crop.txt"
    is_load = True
    load_path = '/media/E/syliu/exp8/ce-lf1.pkl'
    save_path = '/media/E/syliu/exp8/ce-lf2.pkl'

if (dev == "computer"):
    if dataset == 'c':
        train_txt = "/media/xuke/Files/Final/DataSet/list_attr_celeba_train.txt"
    elif dataset == 'l':
        train_txt = "lfwa/train_crop.txt"

    img_root = '/media/xuke/Files/Final/DataSet/train/'
    # val_img_root = img_root
    # val_l_img_root = '/media/liusiyu/data/lfw/lfw-deepfunneled'
    val_c_img_root = '/media/xuke/Files/Final/DataSet/valid/'
    # img_root = '/media/E/syliu/lfw-deepfunneled/'

    #
    test_c_img_root = "/media/xuke/Files/Final/DataSet/test/"
    test_c_txt = "/media/xuke/Files/Final/DataSet/list_attr_celeba_test.txt"
    # val_txt = "val_crop.txt"
    # val_l_txt = "/media/liusiyu/data/lfw/lfwa/test_crop.txt"
    val_c_txt = "/media/xuke/Files/Final/DataSet/list_attr_celeba_val.txt"
    # test_txt = "test_crop.txt"
    # test_txt = "lfwa/test_crop.txt"
    is_load = 0
    # 0 make 1 load 2 transfer
    load_path = '/media/xuke/Files/Final/face attribute/Models/model.pkl'
    save_path = '/media/xuke/Files/Final/face attribute/Models/model.pkl'


def default_loader(path):
    try:
        img = Image.open(path)

        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))


class myDataset(Data.DataLoader):
    def __init__(self, img_dir, img_txt, transform, loader=default_loader):
        img_list = []
        img_labels = []

        fp = open(img_txt)
        # for line in fp.readlines():
        line = fp.readline()
        while line:
            if len(line.split()) != 41:
                line = fp.readline()
                # print(len(line.split()))
                continue
            # print(len(line.split()))
            img_list.append(line.split()[0])
            # img_label_single = []
            # for value in line.split()[1:]:
            #     if value == '-1':
            #         img_label_single.append(0)
            #     if value == '1':
            #         img_label_single.append(1)
            # img_labels.append(img_label_single)
            line = line.split()[1:]
            img_labels.append([int(x == '1') for x in line])
            line = fp.readline()
        self.imgs = [os.path.join(img_dir, file) for file in img_list]
        self.labels = img_labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.int64))
        img = self.loader(img_path)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Cannot transform image: {}'.format(img_path))
        return img, label


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = myDataset(img_dir=img_root,
                          img_txt=train_txt,
                          transform=transform)
train_dataloader = Data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
#drop_last


def show_time(since):
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def make_conv2d(in_chs, out_chs, size, stride, padding, mean, std, groups):
    model = torch.nn.Sequential(
        nn.Conv2d(in_chs, out_chs, size, stride, padding, groups=groups),
        nn.BatchNorm2d(out_chs), nn.PReLU())
    # nn.Conv2d(in_chs, out_chs, size, stride, padding, groups=groups)
    # torch.nn.init.normal_(conv.weight, mean=mean, std=std)
    # return ([conv, nn.BatchNorm2d(out_chs), nn.PReLU()])
    return model


'''
def make_shared_conv(in_channels=3):
    layers = []
    # in_channels = 3
    # conv1
    layers += make_conv2d(in_channels, 96, size=5, stride=2, padding=1, mean=0.0, std=0.01, groups=1)
    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
    # conv2
    layers += make_conv2d(96, 256, size=5, stride=1, padding=2, mean=0.1, std=0.01, groups=2)
    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
    # conv3
    layers += make_conv2d(256, 384, size=3, stride=1, padding=1, mean=0.0, std=0.01, groups=1)
    # conv4
    layers += make_conv2d(384, 384, size=3, stride=1, padding=1, mean=0.1, std=0.01, groups=2)
    # conv5
    layers += make_conv2d(384, 256, size=3, stride=1, padding=1, mean=0.1, std=0.01, groups=2)
    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
    return nn.Sequential(*layers)
'''


def make_shared_conv(in_channels=3):
    layers = []
    # in_channels = 3
    # conv1
    layers += make_conv2d(in_channels,
                          96,
                          size=5,
                          stride=2,
                          padding=1,
                          mean=0.0,
                          std=0.01,
                          groups=1)
    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
    # conv2
    layers += make_conv2d(96,
                          256,
                          size=5,
                          stride=1,
                          padding=2,
                          mean=0.1,
                          std=0.01,
                          groups=2)
    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
    # conv3
    layers += make_conv2d(256,
                          384,
                          size=3,
                          stride=1,
                          padding=1,
                          mean=0.0,
                          std=0.01,
                          groups=1)
    # conv4
    layers += make_conv2d(384,
                          384,
                          size=3,
                          stride=1,
                          padding=1,
                          mean=0.1,
                          std=0.01,
                          groups=2)
    # conv5
    layers += make_conv2d(384,
                          256,
                          size=3,
                          stride=1,
                          padding=1,
                          mean=0.1,
                          std=0.01,
                          groups=2)
    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
    return nn.Sequential(*layers)


def make_shared_fc():
    model = nn.Sequential(
        nn.Linear(256 * 13 * 13, 4096),
        nn.PReLU(),
        nn.Linear(4096, 4096),
        nn.PReLU(),
        nn.Linear(4096, 80),
    )

    return model


class face_attr(nn.Module):
    def __init__(self):
        super(face_attr, self).__init__()
        self.shared_layer = make_shared_conv(in_channels=3)
        self.shared_layer2 = make_shared_fc()
        self._init_weight()

    def forward(self, x):
        self.padding_mode = 'none'
        out_list = []
        # out0
        out = self.shared_layer(x)
        out = out.view(out.size(0), -1)
        out = self.shared_layer2(out)
        # print(out.size())
        out = out.view(-1, 2, 40)
        # print(out.size())
        # for i in range(40):
        # #out = out.permute(1,0,2)
        #     #print(out.size())
        #     out_list.append(out[:, i, :])

        # return out_list
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()
                m.bias.data.zero_()


if is_load == 1:
    module = torch.load(load_path, map_location=torch.device('cuda:0'))
    print("loading")
elif is_load == 0:
    print("making")
    module = face_attr()
elif is_load == 2:

    module = face_attr()
    module_dict = module.state_dict()
    pretrained = torch.load(load_path, map_location=torch.device('cuda:0'))
    pretrained_dict = pretrained.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in module_dict
    }
    module_dict.update(pretrained_dict)
    module.load_state_dict(module_dict)
    print("transfering")
#

# if torch.cuda.device_count() > 1:
#  module = nn.DataParallel(module)
module = module.to(device)
# print(module)

optimizer = optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

loss_list = []

# val_l_dataset = myDataset(img_dir=val_l_img_root,
#                           img_txt=val_l_txt,
#                           transform=transform)
# val_l_dataloader = Data.DataLoader(val_l_dataset,
#                                    batch_size=test_batch_size,
#                                    shuffle=True)

val_c_dataset = myDataset(img_dir=val_c_img_root,
                          img_txt=val_c_txt,
                          transform=transform)
val_c_dataloader = Data.DataLoader(val_c_dataset,
                                   batch_size=test_batch_size,
                                   shuffle=True)

test_c_dataset = myDataset(img_dir=test_c_img_root,
                           img_txt=test_c_txt,
                           transform=transform)
test_c_dataloader = Data.DataLoader(test_c_dataset,
                                    batch_size=test_batch_size,
                                    shuffle=True)

loss_func = nn.CrossEntropyLoss()
# loss_func = nn.BCEWithLogitsLoss()

since = time.time()
# best_acc = 0.0
num_classes = 40
change = 0
weight = [1.0] * 40

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=step_size,
                                            gamma=step_gamma)

for Epoch in range(epoc):
    if stage == 2:
        scheduler.step()
    if Epoch == change_st and stage == 1:
        stage = 2
    module.train()
    all_correct_num = 0
    running_loss = 0.0
    for ii, (img, label) in enumerate(train_dataloader):
        # if Epoch == 0:
        # print(ii)
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        output = module(img)
        loss = 0.0
        loss_attr = [0.0 for i in range(40)]
        # print(loss_attr)
        # print(output.size())
        # print(label.size())

        for i in range(40):
            if stage == 2:
                loss_attr[i] += loss_func(output[:, :, i],
                                          label[:, i]) * weight[i]
            else:
                loss_attr[i] += loss_func(output[:, :, i], label[:, i])
            loss += loss_attr[i]
        # loss = loss_func(output, label)

        loss.backward()
        optimizer.step()
    # for i in range(40):
    #     print("{:.4f}".format(loss_attr[i]))
    # print(tuple(loss_attr))
    max_loss = max(loss_attr)
    min_loss = min(loss_attr)
    avg_loss = sum(loss_attr) / len(loss_attr)
    for i in range(40):

        weight[i] = math.exp((loss_attr[i] - avg_loss) / (max_loss - min_loss))
        # weight[i]=1

    module.eval()
    train_total = 0.0
    train_correct = 0.0
    with torch.no_grad():
        # for index, data in enumerate(train_dataloader):
        for ii, data in enumerate(train_dataloader):
            img, label = data
            img = Variable(img)
            label = Variable(label)
            img = img.to(device)
            label = label.to(device)

            output = module(img)
            _, pred = torch.max(output, 1)
            train_correct += torch.sum(
                pred == label,
                dtype=torch.float32).item()  # num_classes need you to define

            train_total += img.size(0)
    cls_train_Accuracy = train_correct / train_total / 40.0
    # print(cls_train_Accuracy.size())
    mean_Accuracy = cls_train_Accuracy
    print('Epoch ={}'.format(Epoch))
    print('lr={},Loss={:.4f}'.format(
        optimizer.state_dict()['param_groups'][0]['lr'], loss))
    print('train mean Accuracy = {:.4f}'.format(mean_Accuracy))

    show_time(since)

    # val now
    # celeba validation
    if v_dataset == 'c':
        val_dataloader = val_c_dataloader
    else:
        # val_dataloader = val_l_dataloader
        pass
    test_correct = 0.0
    test_total = 0.0
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    module.eval()
    with torch.no_grad():
        for ii, data in enumerate(val_dataloader):
            img, label = data
            img = Variable(img)
            label = Variable(label)
            img = img.to(device)
            label = label.to(device)
            output = module(img)

            _, pred = torch.max(output, 1)

            conf = (confusion_matrix(
                label.view(-1).cpu().numpy(),
                pred.view(-1).cpu().numpy())).ravel()

            tn = tn + conf[0]
            fp = fp + conf[1]
            fn = fn + conf[2]
            tp = tp + conf[3]

            test_correct += torch.sum(
                pred == label,
                dtype=torch.float32).item()  # num_classes need you to define
            test_total += img.size(0)
            # print(image.size(0))
        cls_test_Accuracy = test_correct / test_total / 40.0
        val_mean_Accuracy = cls_test_Accuracy
        print('celeba val mean Accuracy {:.4f}'.format(val_mean_Accuracy))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f2 = 5 * p * r / (4 * p + r)
        print('f2_score = {:.4f}'.format(f2))
        torch.save(module, save_path)
        '''
                if val_mean_Accuracy > best_c_acc:
            torch.save(module, save_path)
            best_c_acc = val_mean_Accuracy
            print("new celeba record ")
            change = 0
        else:
            print('best celeba acc = {:.4f}.'.format(best_c_acc))
            change = change + 1
        '''

        show_time(since)
# test now
# celeba test
    if t_dataset == 'c':
        test_dataloader = test_c_dataloader
    else:
        # test_dataloader = val_l_dataloader
        pass
    test_correct = 0.0
    test_total = 0.0
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    module.eval()
    with torch.no_grad():
        for ii, data in enumerate(test_dataloader):
            img, label = data
            img = Variable(img)
            label = Variable(label)
            img = img.to(device)
            label = label.to(device)
            output = module(img)

            _, pred = torch.max(output, 1)

            conf = (confusion_matrix(
                label.view(-1).cpu().numpy(),
                pred.view(-1).cpu().numpy())).ravel()

            tn = tn + conf[0]
            fp = fp + conf[1]
            fn = fn + conf[2]
            tp = tp + conf[3]

            test_correct += torch.sum(
                pred == label,
                dtype=torch.float32).item()  # num_classes need you to define
            test_total += img.size(0)
            # print(image.size(0))
        cls_test_Accuracy = test_correct / test_total / 40.0
        val_mean_Accuracy = cls_test_Accuracy
        print('celeba test mean Accuracy {:.4f}'.format(val_mean_Accuracy))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f2 = 5 * p * r / (4 * p + r)
        print('f2_score = {:.4f}'.format(f2))

        show_time(since)
'''
# lfwa validation
    test_correct = 0.0
    test_total = 0.0
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    module.eval()
    with torch.no_grad():
        for ii, data in enumerate(val_l_dataloader):
            img, label = data
            img = Variable(img)
            label = Variable(label)
            img = img.to(device)
            label = label.to(device)
            output = module(img)

            _, pred = torch.max(output, 1)

            conf = (confusion_matrix(label.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())).ravel()

            tn = tn + conf[0]
            fp = fp + conf[1]
            fn = fn + conf[2]
            tp = tp + conf[3]

            test_correct += torch.sum(pred == label,dtype=torch.float32).item()  # num_classes need you to define
            test_total += img.size(0)
            # print(image.size(0))
        cls_test_Accuracy = test_correct / test_total / 40.0
        val_mean_Accuracy = cls_test_Accuracy
        print('lfwa val mean Accuracy {:.4f}'.format(val_mean_Accuracy))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f2 = 5 * p * r / (4 * p + r)
        # print('f2_score = {:.4f}'.format(f2))
        # print('{} {} {} {}'.format(tp, tn, fp, fn))
        if val_mean_Accuracy > best_l_acc:
            # torch.save(module, save_path)
            best_l_acc = val_mean_Accuracy
            # print("new lfwa record ")
            change = 0
        else:
            # print('best lfwa acc = {:.4f}.'.format(best_l_acc))
            change = change + 1

        show_time(since)

'''