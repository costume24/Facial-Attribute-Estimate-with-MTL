import torch
import torch.nn as nn

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

def conv_3x3_bn_prelu(inp, oup, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup), nn.PReLU())


def conv_1x1_bn_prelu(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                         nn.BatchNorm2d(oup), nn.PReLU())

class psnet(nn.Module):
    def __init__(self,
                 use_1x1=True,
                 prelu='no',
                 ratio=0.25,
                 num_attributes=40,
                 input_size=224):
        super().__init__()
        self.use_1x1 = use_1x1
        self.pool = nn.MaxPool2d(2, 2)
        if prelu == 'yes':
            conv3 = conv_3x3_bn_prelu
            conv1 = conv_1x1_bn_prelu
        else:
            conv3 = conv_3x3_bn
            conv1 = conv_1x1_bn
        self.t_conv = nn.ModuleList([
            conv3(3, 32),
            conv3(64, 64),
            conv3(128, 128),
            conv3(256, 256),
            conv3(512, 128)
        ])
        self.s_conv = nn.ModuleList([
            conv3(3, 32),
            conv3(64, 64),
            conv3(96, 128),
            conv3(160, 256),
            conv3(288, 128)
        ])  # (5,),s支路的5个卷积层
        self.t_fc = nn.ModuleList([nn.Linear(7680, 1024), nn.Linear(1024, 512)])
        self.s_fc = nn.ModuleList([nn.Linear(4800, 1024),
                                   nn.Linear(1024, 512)])  # (2,)，s支路的2个FC层
        self.output = []  # (4,), 4个支路的输出
        self.conv_1x1 = nn.ModuleList([
            conv1(32, 32),
            conv1(64, 32),
            conv1(128, 32),
            conv1(256, 32),
            conv1(128,32)
        ])
        self.group = nn.ModuleList([
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512)
        ])
        self.group.append(
            nn.ModuleList([
                nn.Linear(512, 26),
                nn.Linear(512, 12),
                nn.Linear(512, 18),
                nn.Linear(512, 24)
            ]))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        self.output = [0] * 4
        block_1, s_1 = self.block(input, input, 0)

        block_2, s_2 = self.block(block_1, s_1, 1)

        block_3, s_3 = self.block(block_2, s_2, 2)

        block_4, s_4 = self.block(block_3, s_3, 3)

        block_5, s_5 = self.block(block_4, s_4, 4)

        block_5 = block_5.view(-1, block_5.size()[1] * block_5.size()[2] * block_5.size()[3])
        s_5 = s_5.view(-1, s_5.size()[1] * s_5.size()[2] * s_5.size()[3])

        block_5 = self.t_fc[0](block_5)
        # s_0_fc1 = self.s_fc[0](s_5)

        block_5 = self.t_fc[1](block_5)
        # s_0_fc2 = self.s_fc[1](s_0_fc1)

        for i in range(4):
            self.output[i] = self.group[0][i](block_5)
        for i in range(4):
            self.output[i] = self.group[1][i](self.output[i])
        output_0, output_1, output_2, output_3 = self.output
        return output_0, output_1, output_2, output_3

    def block(self, t_0, s_0, ind):
        t_0 = self.t_conv[ind](t_0)

        t_0 = self.pool(t_0)

        s_0 = self.s_conv[ind](s_0)

        s_0 = self.pool(s_0)

        if not self.use_1x1:
            indices = torch.arange(0, 32, 1).cuda()

            t_0_partial = torch.index_select(t_0, 1, indices).cuda()

            t_0 = torch.cat([t_0, s_0], 1)

            s_0 = torch.cat([t_0_partial, s_0], 1)
        else:
            t_0_1x1 = self.conv_1x1[ind](t_0)

            t_0 = torch.cat([t_0, s_0], 1)

            s_0 = torch.cat([t_0_1x1, s_0], 1)
        return t_0, s_0


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)