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
        self.t_conv = nn.ModuleList([
                conv_3x3_bn(3, 32),
                conv_3x3_bn(32, 64),
                conv_3x3_bn(64, 128),
                conv_3x3_bn(128, 256),
                conv_3x3_bn(256, 128)
            ])  
        self.t_fc = nn.ModuleList([nn.Linear(3840, 512), nn.Linear(512, 512)])
        self.output = []  # (4,), 4个支路的输出

        self.group = nn.ModuleList([
            nn.Linear(512, 26),
            nn.Linear(512, 12),
            nn.Linear(512, 18),
            nn.Linear(512, 24)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        self.output = [0] *4
        block_1 = self.block(input, 0)

        block_2= self.block(block_1, 1)

        block_3 = self.block(block_2, 2)

        block_4 = self.block(block_3, 3)

        block_5 = self.block(block_4, 4)

        block_5 = block_5.view(-1, block_5.size()[1] * block_5.size()[2] * block_5.size()[3])

        block_5 = self.t_fc[0](block_5)

        block_5 = self.t_fc[1](block_5)

        for i in range(4):
            self.output[i] = self.group[i](block_5)
        output_0, output_1, output_2, output_3 = self.output
        return output_0, output_1, output_2, output_3

    def block(self, inp, ind):
        inp = self.t_conv[ind](inp)
        t_0 = self.pool(inp)
        return t_0


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