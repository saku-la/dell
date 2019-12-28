
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)



class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

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

class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)

class ldyNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()
        self.upconv=upconv(num_classes,num_classes,3,2)
        self.initial_block1 = DownsamplerBlock(1,16)

        self.initial_block2 =(DownsamplerBlock(16,64))

        self.layers1 = nn.ModuleList()

        for x in range(0, 5):    #5 times
           self.layers1.append(non_bottleneck_1d(64, 0.1, 1))  

        self.initial_block3 = (DownsamplerBlock(64,128))

        self.layers2 = nn.ModuleList()

        for x in range(0, 2):    #2 times
            self.layers2.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers2.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers2.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers2.append(non_bottleneck_1d(128, 0.1, 16))

        #only for encoder mode:
        self.output_conv1 = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)


        self.initial_block4 = (UpsamplerBlock(128,64))

        self.layers3 = nn.ModuleList()
        self.layers3.append(non_bottleneck_1d(64, 0, 1))
        self.layers3.append(non_bottleneck_1d(64, 0, 1))

        self.initial_block5 = (UpsamplerBlock(64,16))

        self.layers4 = nn.ModuleList()
        self.layers4.append(non_bottleneck_1d(16, 0, 1))
        self.layers4.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv2 = nn.ConvTranspose2d( 16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        input=self.upconv(input)
        input=self.upconv(input)
        output = self.initial_block1(input)

        x1=output

        output = self.initial_block2(output)

        x2=output
        for layer in self.layers1:
            output = layer(output)
        x3=output

        output = self.initial_block3(output)
       
        #x4=output
        for layer in self.layers2:
            output = layer(output)
        
        output = self.initial_block4(output)

        output=x2+output
        for layer in self.layers3:
            output = layer(output)
        #output = output + x3
        output = self.initial_block5(output)
        #output=output+x1
        for layer in self.layers4:
            output = layer(output)
        #print('vvvvvvvvvvvvvvvv',output.shape)
        output = self.output_conv2(output)

        return output
