
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from ASPP import ASPP
from seg_opr.seg_oprs import ConvBnRelu
from collections import OrderedDict
import numpy as np
class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock,self).__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        # print("self.conv(input)",self.conv(input).shape)
        # print("self.pool(input)",self.pool(input).shape)
        
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        
        output = self.bn(output)
        return F.relu(output)
    
# grads = {}

# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super(non_bottleneck_1d,self).__init__()

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

class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(
                F.interpolate(pooling(x), size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super(Encoder,self).__init__()
        self.initial_block = DownsamplerBlock(1,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.1, 1))  

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        #only for encoder mode:
        self.output_conv = nn.Conv2d(128, 256, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)
            # print('ooooooooooooooooooooooooo')
            
        
        return output
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, channel, reduct_ratio=16):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([("avgpool", nn.AdaptiveAvgPool2d(1)),
                                                     ("linear1", nn.Conv2d(channel, channel // reduct_ratio,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("relu", nn.ReLU(inplace=True)),
                                                     ("linear2", nn.Conv2d(channel // reduct_ratio, channel,
                                                                           kernel_size=1, stride=1, padding=0))]))

    def forward(self, x):
        inputs = x
        chn_se = self.channel_se(x).sigmoid().exp()
        return torch.mul(inputs, chn_se)

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

class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super(UpsamplerBlock,self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super(Decoder,self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)#fan juan ji

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

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
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)

class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        #print(num_in_layers)
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        #print(x)
        x = self.normalize(x)
        #print(x)
        return 0.3 * self.relu(x)

class ERFNet(nn.Module):
    def __init__(self, num_classes):  #use encoder to pass pretrained encoder
        super(ERFNet,self).__init__()

        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(1)
        self.aspp = ASPP(dim_in=128,dim_out=128,rate=1,bn_mom=0.007)
        self.psp_layer = PyramidPooling('psp', 128, 128,
                                        norm_layer=nn.BatchNorm2d)
    def forward(self, input):
        # a6=input
        output = self.encoder(input)    #predict=False by default
        #output = self.aspp(output)
        # output = self.psp_layer(output)# duib shiyan jiezhujicu
        # print('xxxxxxxxxxxxxxxxxxxxxx',output.shape)
        # if(output()>0.5):
        #     output=1
        # if(output()<=0.5):
        #     output=0
        # #############################################
        # output=output.split(1,1)
        # disp_show = output[100].squeeze()
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(disp_show.data.cpu().numpy())
        # plt.show() 
        # #############################################
        
        output=self.decoder.forward(output)
        output[output>0.5]=1
        output[output<0.6]=0
        output=torch.mul(output,input)
        # print("output",output)
        # print("input",input)
        return output
class ERFNet2(nn.Module):
    def __init__(self, num_classes):  #use encoder to pass pretrained encoder
        super(ERFNet2,self).__init__()

        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(1)
        self.aspp = ASPP(dim_in=128,dim_out=128,rate=1,bn_mom=0.007)
        self.psp_layer = PyramidPooling('psp', 128, 128,
                                        norm_layer=nn.BatchNorm2d)
    def forward(self, input):
        # a6=input
        output = self.encoder(input)    #predict=False by default
        #output = self.aspp(output)
        # output = self.psp_layer(output)# duib shiyan jiezhujicu
        # print('xxxxxxxxxxxxxxxxxxxxxx',output.shape)
        # if(output()>0.5):
        #     output=1
        # if(output()<=0.5):
        #     output=0
        # #############################################
        # output=output.split(1,1)
        # disp_show = output[100].squeeze()
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(disp_show.data.cpu().numpy())
        # plt.show() 
        # #############################################
        
        output=self.decoder.forward(output)
        output[output>0.5]=1
        output[output<0.6]=0
        output=torch.mul(output,input)
        # print("output",output)
        # print("input",input)
        return output

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
        # a6=x1
        # print('out',x1.grad)
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

class Resnet50_md(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet50_md, self).__init__()
        # encoder
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock(64, 64, 3, 2)  # H/8  -  256D
        self.conv3 = resblock(256, 128, 4, 2)  # H/16 -  512D
        self.conv4 = resblock(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5 = resblock(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.upconv6 = upconv(2048, 512, 3, 2)
        self.iconv6 = conv(1024 + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(256+128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64+2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(32+64+2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        #print(x.shape)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        #print(iconv4.shape)
        self.disp4 = self.disp4_layer(iconv4)
        
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        aaaaa=self.disp1
        # aaaaa.reshape([1,1,256,256])
        return aaaaa



class JsyNet(nn.Module):
    def __init__(self, num_classes):  #use encoder to pass pretrained encoder
        super(JsyNet,self).__init__()

        self.ERFNet = ERFNet(num_classes)
        self.ERFNet2 = ERFNet2(num_classes)
        # self.aspp = ASPP(dim_in=128,dim_out=128,rate=1,bn_mom=0.007)
        # self.psp_layer = PyramidPooling('psp', 128, 128,
        #                                 norm_layer=nn.BatchNorm2d)
    def forward(self, input):

        output= self.ERFNet(input)    #predict=False by default
        #output = self.aspp(output)
        # output = self.psp_layer(output)# duib shiyan jiezhujicu
        # print('xxxxxxxxxxxxxxxxxxxxxx',output.shape)
        
        # i=0
        # left1=input
        # while i<15:
                    
        #     left1=torch.cat((left1,input),1)
        #     i=i+1
        # Net2Iput=output.mul(left1)#点乘
        # # print(Net2Iput.grad)
        # Net2Iput11,Net2Iput12=Net2Iput.split(128,2)
        # Net2Iput11,Net2Iput21=Net2Iput11.split(128,3)
        # Net2Iput12,Net2Iput22=Net2Iput12.split(128,3)
                
        #         # Net2Iput1=Net2Iput[:127]
        #         # Net2Iput2=Net2Iput[128:]
        #         # Net2Iput11=Net2Iput1[:,0:127]
        #         # Net2Iput12=Net2Iput1[:,128:255]
        #         # Net2Iput21=Net2Iput2[:,0:127]
        #         # Net2Iput22=Net2Iput2[:,128:255]
                
        # Net2Iput11=torch.sum(Net2Iput11,2,keepdim=True, out=None)
        # Net2Iput11=torch.sum(Net2Iput11,3,keepdim=True, out=None)
        # Net2Iput12=torch.sum(Net2Iput12,2,keepdim=True, out=None)
        # Net2Iput12=torch.sum(Net2Iput12,3,keepdim=True, out=None)
        # Net2Iput21=torch.sum(Net2Iput21,2,keepdim=True, out=None)
        # Net2Iput21=torch.sum(Net2Iput21,3,keepdim=True, out=None)
        # Net2Iput22=torch.sum(Net2Iput22,2,keepdim=True, out=None)
        # Net2Iput22=torch.sum(Net2Iput22,3,keepdim=True, out=None)
        # Net2Iput1=torch.cat((Net2Iput11,Net2Iput12),2)
        # Net2Iput2=torch.cat((Net2Iput21,Net2Iput22),2)
        # Net2Iput=torch.cat((Net2Iput1,Net2Iput2),3)
        #         # Net2Iput=torch.cat((Net2Iput,Net2Iput22),1)
        # ps = nn.PixelShuffle(4)
        # Net2Iput = ps(Net2Iput)

        Net2Iput=self.ERFNet2(output)
        # output[output>0.5]=1
        # output[output<0.6]=0
        
        return Net2Iput