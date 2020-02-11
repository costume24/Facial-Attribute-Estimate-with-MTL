import torch
import torch.nn as nn

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class psnet(nn.Module):
    def __init__(self, ratio=0.25, num_attributes=40, input_size=224):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = conv_3x3_bn(3, 32)
        self.t_conv2 = conv_3x3_bn(192, 64)
        self.t_conv3 = conv_3x3_bn(256, 128)
        self.t_conv4 = conv_3x3_bn(384, 256)
        self.t_conv5 = conv_3x3_bn(640, 128)

        self.s_conv1 = conv_3x3_bn(3, 32)
        self.s_conv2 = conv_3x3_bn(160, 64)
        self.s_conv3 = conv_3x3_bn(192, 128)
        self.s_conv4 = conv_3x3_bn(256, 256)
        self.s_conv5 = conv_3x3_bn(384, 128)

        self.t_fc1 = nn.Linear(11520, 512)
        self.t_fc2 = nn.Linear(512, 512)

        self.s_fc1 = nn.Linear(7680, 512)
        self.s_fc2 = nn.Linear(512, 512)

        # 这里的四个分支的输出维度由手工分组决定
        # 现在采用论文中的分组方式
        # 暂时写死，后续可以改为由输入参数决定
        self.group1 = nn.Linear(1024, 26)
        self.group2 = nn.Linear(1024, 12)
        self.group3 = nn.Linear(1024, 18)
        self.group4 = nn.Linear(1024, 24)

    def forward(self, input):
        # print('input size ', input.size())
        t_0_1, t_1_1, t_2_1, t_3_1, s_1 = self.block(input, input, input, input, input, self.t_conv1, self.s_conv1)
        # print('After 1 block: ')
        # print('t_0_1 :', t_0_1.size())
        # print('t_1_1 :', t_1_1.size())
        # print('t_2_1 :', t_2_1.size())
        # print('t_3_1 :', t_3_1.size())
        # print('s_1 :', s_1.size())

        t_0_2, t_1_2, t_2_2, t_3_2, s_2 = self.block(t_0_1, t_1_1, t_2_1, t_3_1, s_1, self.t_conv2, self.s_conv2)
        # print('After 2 block: ')
        # print('t_0_2 :', t_0_2.size())
        # print('t_1_2 :', t_1_2.size())
        # print('t_2_2 :', t_2_2.size())
        # print('t_3_2 :', t_3_2.size())
        # print('s_2 :', s_2.size())

        t_0_3, t_1_3, t_2_3, t_3_3, s_3 = self.block(t_0_2, t_1_2, t_2_2, t_3_2, s_2, self.t_conv3, self.s_conv3)
        # print('After 3 block: ')
        # print('t_0_3 :', t_0_3.size())
        # print('t_1_3 :', t_1_3.size())
        # print('t_2_3 :', t_2_3.size())
        # print('t_3_3 :', t_3_3.size())
        # print('s_3 :', s_3.size())
        t_0_4, t_1_4, t_2_4, t_3_4, s_4 = self.block(t_0_3, t_1_3, t_2_3, t_3_3, s_3, self.t_conv4, self.s_conv4)
        # print('After 4 block: ')
        # print('t_0_4 :', t_0_4.size())
        # print('t_1_4 :', t_1_4.size())
        # print('t_2_4 :', t_2_4.size())
        # print('t_3_4 :', t_3_4.size())
        # print('s_4 :', s_4.size())
        t_0_5, t_1_5, t_2_5, t_3_5, s_5 = self.block(t_0_4, t_1_4, t_2_4, t_3_4, s_4, self.t_conv5, self.s_conv5)
        # print('After 5 block: ')
        # print('t_0_5 :', t_0_5.size())
        # print('t_1_5 :', t_1_5.size())
        # print('t_2_5 :', t_2_5.size())
        # print('t_3_5 :', t_3_5.size())
        # print('s_5 :', s_5.size())

        t_0_5 = t_0_5.view(-1, t_0_5.size()[1] * t_0_5.size()[2] * t_0_5.size()[3])
        t_1_5 = t_1_5.view(-1, t_1_5.size()[1] * t_1_5.size()[2] * t_1_5.size()[3])
        t_2_5 = t_2_5.view(-1, t_2_5.size()[1] * t_2_5.size()[2] * t_2_5.size()[3])
        t_3_5 = t_3_5.view(-1, t_3_5.size()[1] * t_3_5.size()[2] * t_3_5.size()[3])
        s_5 = s_5.view(-1, s_5.size()[1] * s_5.size()[2] * s_5.size()[3])

        t_0_fc1 = self.t_fc1(t_0_5)
        t_1_fc1 = self.t_fc1(t_1_5)
        t_2_fc1 = self.t_fc1(t_2_5)
        t_3_fc1 = self.t_fc1(t_3_5)
        s_0_fc1 = self.s_fc1(s_5)

        t_0_fc2 = self.t_fc2(t_0_fc1)
        t_1_fc2 = self.t_fc2(t_1_fc1)
        t_2_fc2 = self.t_fc2(t_2_fc1)
        t_3_fc2 = self.t_fc2(t_3_fc1)
        s_0_fc2 = self.s_fc2(s_0_fc1)

        output_0 = torch.cat([t_0_fc2, s_0_fc2], 1)
        output_1 = torch.cat([t_1_fc2, s_0_fc2], 1)
        output_2 = torch.cat([t_2_fc2, s_0_fc2], 1)
        output_3 = torch.cat([t_3_fc2, s_0_fc2], 1)
        output_4 = torch.cat([s_0_fc2, s_0_fc2], 1)

        output_0 = self.group1(output_0)
        output_1 = self.group2(output_1)
        output_2 = self.group3(output_2)
        output_3 = self.group4(output_3)

        return output_0, output_1, output_2, output_3
        # logit = torch.cat([output_0, output_1, output_2, output_3], 1)

        # return logit

    def block(self, t_0, t_1, t_2, t_3, s_0, tconv, sconv):
        t_0 = tconv(t_0)
        t_1 = tconv(t_1)
        t_2 = tconv(t_2)
        t_3 = tconv(t_3)

        t_0 = self.pool(t_0)
        t_1 = self.pool(t_1)
        t_2 = self.pool(t_2)
        t_3 = self.pool(t_3)

        indices = torch.arange(0, 32, 1).cuda()
        t_0_partial = torch.index_select(t_0, 1, indices).cuda()
        t_1_partial = torch.index_select(t_1, 1, indices).cuda()
        t_2_partial = torch.index_select(t_2, 1, indices).cuda()
        t_3_partial = torch.index_select(t_3, 1, indices).cuda()
        s_0 = sconv(s_0)
        s_0 = self.pool(s_0)
        s_0 = torch.cat([t_0_partial, t_1_partial, t_2_partial, t_3_partial, s_0], 1)

        t_0 = torch.cat([t_0, s_0], 1)
        t_1 = torch.cat([t_1, s_0], 1)
        t_2 = torch.cat([t_2, s_0], 1)
        t_3 = torch.cat([t_3, s_0], 1)

        return t_0, t_1, t_2, t_3, s_0