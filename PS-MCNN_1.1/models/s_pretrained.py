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
        conv3 = conv_3x3_bn
        conv1 = conv_1x1_bn
        self.s_conv = nn.ModuleList([
            conv3(3, 32),
            conv3(64, 64),
            conv3(96, 128),
            conv3(160, 256),
            conv3(288, 128)
        ])  # (5,),s支路的5个卷积层
        self.s_fc = nn.ModuleList([nn.Linear(4800, 1024),
                                   nn.Linear(1024, 10177)])  # (2,)，s支路的2个FC层

        self.conv_1x1 = nn.ModuleList([
            conv1(32, 64),
            conv1(64, 96),
            conv1(128, 160),
            conv1(256, 288),
            conv1(128,160)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        s_1 = self.block(input, 0)

        s_2 = self.block( s_1, 1)

        s_3 = self.block( s_2, 2)

        s_4 = self.block( s_3, 3)

        s_5 = self.block( s_4, 4)

        s_5 = s_5.view(-1, s_5.size()[1] * s_5.size()[2] * s_5.size()[3])

        s_0_fc1 = self.s_fc[0](s_5)

        s_0_fc2 = self.s_fc[1](s_0_fc1)

        return s_0_fc2

    def block(self, s_0, ind):
        s_0 = self.s_conv[ind](s_0)
        s_0 = self.pool(s_0)
        s_0 = self.conv_1x1[ind](s_0)
        return s_0