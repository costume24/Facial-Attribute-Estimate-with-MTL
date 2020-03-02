import torch
import torch.nn as nn

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
class psnet(nn.Module):
    def __init__(self,
                 use_1x1=True,
                 ratio=0.25,
                 num_attributes=40,
                 input_size=224):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv = nn.ModuleList()  # (4,5),每一行是一个t支路的5个卷积层
        self.t_fc = nn.ModuleList()  # (4,2)，每一行是一个t支路的2个FC层
        self.output = []  # (4,), 4个支路的输出
        for _ in range(4):
            tmp = nn.ModuleList([
                conv_3x3_bn(3, 32),
                conv_3x3_bn(32, 64),
                conv_3x3_bn(64, 128),
                conv_3x3_bn(128, 256),
                conv_3x3_bn(256, 128)
            ])
            self.t_conv.append(tmp)

        for _ in range(4):
            tmp = nn.ModuleList([nn.Linear(3840, 512), nn.Linear(512, 512)])
            self.t_fc.append(tmp)

        self.group = nn.ModuleList([
            nn.Linear(512, 26),
            nn.Linear(512, 12),
            nn.Linear(512, 18),
            nn.Linear(512, 24)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        self.output = []
        block_1 = self.block([input, input, input, input], 0)

        block_2= self.block(block_1, 1)

        block_3 = self.block(block_2, 2)

        block_4 = self.block(block_3, 3)

        block_5 = self.block(block_4, 4)

        for i in range(4):
            _size = block_5[i].size()
            block_5[i] = block_5[i].view(-1, _size[1] * _size[2] * _size[3])

        for i in range(4):
            block_5[i] = self.t_fc[i][0](block_5[i])

        for i in range(4):
            block_5[i] = self.t_fc[i][1](block_5[i])

        for _ in range(4):
            self.output.append(block_5[i])
        for i in range(4):
            self.output[i] = self.group[i](self.output[i])
        output_0, output_1, output_2, output_3 = self.output
        return output_0, output_1, output_2, output_3

    def block(self, inp, ind):
        t_0, t_1, t_2, t_3 = inp
        t_0 = self.t_conv[0][ind](t_0)
        t_1 = self.t_conv[1][ind](t_1)
        t_2 = self.t_conv[2][ind](t_2)
        t_3 = self.t_conv[3][ind](t_3)

        t_0 = self.pool(t_0)
        t_1 = self.pool(t_1)
        t_2 = self.pool(t_2)
        t_3 = self.pool(t_3)

        return [t_0, t_1, t_2, t_3]


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