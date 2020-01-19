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
import sklearn
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'celeba'
# dataset = 'lfwa'

if(dataset == 'celeba'):
    img_root = '/media/E/syliu/celeba_crop/'
    train_txt = "train_crop_clean.txt"
    val_txt = "val_crop.txt"
    test_txt = "test_crop.txt"
else:
    img_root = '/media/E/syliu/lfw-deepfunneled/'
    train_txt = "lfwa/train_crop.txt"
    val_txt = "lfwa/test_crop.txt"
    test_txt = "lfwa/test_crop.txt"

load_path = '/media/E/syliu/exp4_7.pkl'


batch_size = 300
test_batch_size = batch_size
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
            img_label_single = []
            for value in line.split()[1:]:
                if value == '0':
                    img_label_single.append(0)
                if value == '1':
                    img_label_single.append(1)
            img_labels.append(img_label_single)
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
    transforms.Normalize(mean=[90.1146/256.0, 103.035/256.0, 127.689/256.0],
                         std=[0.5, 0.5, 0.5])
])

def show_time(since):
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def make_conv2d(in_chs, out_chs, size, stride, padding, mean, std, groups):
    model = torch.nn.Sequential(
        nn.Conv2d(in_chs, out_chs, size, stride, padding, groups=groups),
        nn.BatchNorm2d(out_chs),
        nn.PReLU()
    )
    # nn.Conv2d(in_chs, out_chs, size, stride, padding, groups=groups)
    # torch.nn.init.normal_(conv.weight, mean=mean, std=std)
    # return ([conv, nn.BatchNorm2d(out_chs), nn.PReLU()])
    return model


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
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



module = torch.load(load_path, map_location=torch.device('cuda:0'))


#

# if torch.cuda.device_count() > 1:
#  module = nn.DataParallel(module)
module = module.to(device)
# print(module)

# optimizer = optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

loss_list = []

test_dataset = myDataset(img_dir=img_root, img_txt=test_txt, transform=transform)
test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
print(len(test_dataset))
print(len(test_dataloader))

loss_func = nn.CrossEntropyLoss()

since = time.time()
# best_acc = 0.0
num_classes = 40

# test now
test_correct = 0.0
test_total = 0.0
test_correct = torch.zeros(40)
module.eval()
with torch.no_grad():
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    for ii, data in enumerate(test_dataloader):
        # print(data)
        # print(len(data))
        img, label = data
        img = Variable(img)
        label = Variable(label)
        img = img.to(device)
        label = label.to(device)
        output = module(img)

        ths=0.5
        pred = torch.zeros(img.size(0),40)
        pred = pred.to(device)
        # for i in range(ii):
        #     for j in range(40):
        #         if (output[i][0][j] < ths):
        #             pred[i][j] = 1.0
        #
        _, pred = torch.max(output, 1)
        # print(output.size())

        # conf = (sklearn.metrics.confusion_matrix(label.view(-1).cpu().numpy(),pred.view(-1).cpu().numpy())).ravel()
        #
        # tn = tn + conf[0]
        # fp = fp + conf[1]
        # fn = fn + conf[2]
        # tp = tp + conf[3]


        pred = pred.long()
        # print(pred.size())
        # print(label.size())
        for i in range(40):
            test_correct[i] = (test_correct[i] + torch.sum(pred[:,i] == label[:,i], dtype=torch.float32).item())
            # for j in range(pred[:,i].size()):
            #     if (pred[:,i]==1 and label[j,i]==1):tp = tp + 1


          # num_classes need you to define
        # print(test_correct)
        test_total += img.size(0)
        # print(image.size(0))
    # print(test_correct)
    # print('tp{} tn{} fp{} fn{}'.format(tp,tn,fp,fn))
    # p = tp/(tp+fp)
    # r = tp/(tp+fn)
    # f2 = 5 * p * r / (4 * p + r)
    # print('f2_score = {:.4f}'.format(f2))
    cls_test_Accuracy = (test_correct) / test_total
    test_mean_Accuracy = sum(cls_test_Accuracy)/40.0
    print('test mean Accuracy {:.4f}'.format(test_mean_Accuracy))
    # for i in range(40):
    #     print('attr{}:{:.4f}'.format(cls_test_Accuracy[i]))
    print(cls_test_Accuracy)



    show_time(since)