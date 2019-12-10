from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)

class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)




class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers,1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale,):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class Jnet(nn.Module):
    def __init__(self, num_in_layers):
        super(Jnet, self).__init__()

        self.conv1 = conv(num_in_layers, 16, 3, 1)
        self.conv2 = conv(16, 32, 3, 1)
        self.res1=resblock(32, 8, 3, 1)
        self.conv3=nn.Conv2d(32,32,4,2,1)#down
        self.conv4 = conv(128, 64, 3, 1)
        self.conv5 = conv(64, 32, 3, 1)
        self.conv6 = conv(32, 1, 3, 1)
        self.conv8=conv(192,128,3,1)
        self.conv7=nn.Conv2d(64,64,4,2,1)#down
        self.upconv1 = upconv(32, 32, 3, 2)
        self.maxpool=maxpool(3)
    def forward(self,x):
        left = nn.functional.interpolate(x, (1,4096), mode='bilinear', align_corners=True)
                
        #left=left.squeeze(2)
        conv=nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3,1), stride=1,padding=1)
        left=conv(left)
        # left=left.unsqueeze(2)
        left=left.permute(0,3,1,2)
        ps = nn.PixelShuffle(64)
        Net2Iput = ps(left)
        Net2Iput=Net2Iput/40000.0
        x1=self.conv1(Net2Iput)
        x2=self.conv2(x1)

        x31=self.res1(x2)
        x32=self.res1(x31)
        x33=self.res1(x32)
        x4=self.maxpool(x2)
        x41=self.res1(x4)
        x42=self.res1(x41)
        x43=self.res1(x42)
        x5=self.maxpool(x4)
        x51=self.res1(x5)
        x52=self.res1(x51)
        x53=self.res1(x52)
        x6=self.maxpool(x5)
        x61=self.res1(x6)
        x62=self.res1(x61)
        x63=self.res1(x62)
        x7=self.maxpool(x6)
        x71=self.res1(x7)
        x72=self.res1(x71)
        x73=self.res1(x72)
        x8=self.maxpool(x7)
        x81=self.res1(x8)
        x82=self.res1(x81)
        x83=self.res1(x82)

        x8up=self.upconv1(x83)
        x8up=self.upconv1(x8up)
        x8up=self.upconv1(x8up)
        x8up=self.upconv1(x8up)
        x8up=self.upconv1(x8up)
        x7up=self.upconv1(x73)
        x7up=self.upconv1(x7up)
        x7up=self.upconv1(x7up)
        x7up=self.upconv1(x7up)
        x6up=self.upconv1(x63)
        x6up=self.upconv1(x6up)
        x6up=self.upconv1(x6up)
        x5up=self.upconv1(x53)
        x5up=self.upconv1(x5up)
        x4up=self.upconv1(x43)

        concat=torch.cat((x33,x4up),1)
        concat=torch.cat((concat,x5up),1)
        concat=torch.cat((concat,x6up),1)
        concat=torch.cat((concat,x7up),1)
        concat=torch.cat((concat,x8up),1)

        concat=self.conv8(concat)
        concat=self.conv4(concat)
        concat=self.maxpool(concat)
        concat=self.conv5(concat)
        concat=self.conv6(concat)
        
        return concat





