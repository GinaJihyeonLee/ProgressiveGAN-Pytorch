import torch
import torch.nn as nn
import math
import numpy as np

def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, pixel=False, only=False, equalized=True):
    if equalized:
        layers.append(EqualizedConv2d(c_in, c_out, k_size, stride, pad))
    else:
        layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        layers.append(nn.LeakyReLU(0.2))
        if bn: layers.append(nn.BatchNorm2d(c_out))
        elif pixel: layers.append(PixelNorm())
    return layers

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5

class EqualizedConv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad):
        super(EqualizedConv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad)
        self.conv.weight.data.normal_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)*self.equalized_lr()
        return out
    
    def equalized_lr(self):
        layer = self.conv.weight.size()
        fan_in = np.prod(layer[1:])
        return math.sqrt(2.0/fan_in)

class EqualizedLinear(nn.Module):
    def __init__(self, c_in, c_out):
          super(EqualizedLinear, self).__init__()
          self.linear = nn.Linear(c_in, c_out)
          self.linear.weight.data.normal_()
          self.linear.bias.data.zero_()

    def forward(self, x):
        out = self.linear(x)*self.equalized_lr()
        return out
    
    def equalized_lr(self):
        layer = self.linear.weight.size()
        fan_in = np.prod(layer[1:])
        return math.sqrt(2.0/fan_in)

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.main_layers = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        self.main_layers.append(self.init_layers())
        self.to_rgb_layers.append(self.init_rgb())

    def init_layers(self):
        layers=[]
        ndim = self.config.ngf
        layers = conv(layers, self.config.nz, ndim, 4, 1, 3, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
        return nn.Sequential(*layers)

    def init_rgb(self):
        layers=[]
        layers= conv(layers,self.config.ngf,self.config.nc,1,1,0,only=True)
        if self.config.flag_tanh:  
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def stage_up(self, resl):
        assert 3<=resl<=10
        print(f'growing network {pow(2,resl-1)}x{pow(2,resl-1)} to {pow(2,resl)}x{pow(2,resl)}')
        layers, ndim = self.middle_layers(resl)
        self.main_layers.append(layers)
        self.to_rgb_layers.append(self.to_rgb(ndim))

    def to_rgb(self, c_in):
        layers=[]
        layers= conv(layers,c_in,self.config.nc,1,1,0,only=True)
        if self.config.flag_tanh:  
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def middle_layers(self, resl):
        ndim = self.config.ngf
        halving = False
        if resl in range(3,6):
            halving = False
        elif resl in range(6,11):
            halving = True
            for i in range(resl-5):
                ndim=ndim//2
        layers=[]
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if halving:
            layers = conv(layers, ndim*2, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
        return nn.Sequential(*layers), ndim

    def forward(self, x, alpha):
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        x=x.view(x.size(0),-1,1,1)
        for layer in self.main_layers[:-1]:
            x = layer(x)      
        if alpha!=1:
            prev_x = self.to_rgb_layers[-2](upsample(x))
        x = self.to_rgb_layers[-1](self.main_layers[-1](x))
        if alpha!=1:
            x = alpha*x + (1-alpha)*prev_x
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.main_layers = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()
        self.main_layers.append(self.init_layers())
        self.from_rgb_layers.append(self.init_rgb())
        if config.flag_equalized:
            self.linear=EqualizedLinear(config.ndf,1)
        else:
            self.linear=nn.Linear(config.ndf,1)

    def init_layers(self):
        layers=[]
        ndim = self.config.ndf
        layers = conv(layers, ndim+1, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn)
        layers = conv(layers, ndim, ndim, 4, 1, 0, self.config.flag_leaky, self.config.flag_bn)
        return nn.Sequential(*layers)

    def init_rgb(self):
        layers=[]
        layers= conv(layers, self.config.nc, self.config.ndf, 1, 1, 0)
        return nn.Sequential(*layers)

    def from_rgb(self, c_in):
        layers=[]
        layers= conv(layers, self.config.nc, c_in, 1, 1, 0)
        return nn.Sequential(*layers)

    def middle_layers(self, resl):
        ndim = self.config.ndf
        doubling = False
        if resl in range(3,6):
            doubling = False
        elif resl in range(6,11):
            doubling = True
            for i in range(resl-5):
                ndim=ndim//2
        layers=[]
        if doubling:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn)
            layers = conv(layers, ndim, ndim*2, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn)
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn)
        layers.append(nn.AvgPool2d(kernel_size=2))
        return nn.Sequential(*layers), ndim

    def stage_up(self, resl):
        assert 3<=resl<=10
        print(f'growing network {pow(2,resl-1)}x{pow(2,resl-1)} to {pow(2,resl)}x{pow(2,resl)}')
        layers, ndim = self.middle_layers(resl)
        self.from_rgb_layers.append(self.from_rgb(ndim))
        self.main_layers.append(layers)

    def get_batch_std(self, x):
        size = x.shape
        y = x.std([0]).mean().expand(size[0],1,size[2],size[3])
        return y

    def forward(self, x, alpha):
        downsample = nn.AvgPool2d(kernel_size=2)
        if alpha!=1:
          prev_x = self.from_rgb_layers[-2](downsample(x))
        x = self.from_rgb_layers[-1](x)
        for i,layer in enumerate(reversed(self.main_layers[1:])):
            x = layer(x)
            if i==0 and alpha!=1:
                x = alpha*x + (1-alpha)*prev_x
        std = self.get_batch_std(x)
        x = torch.cat((x,std),dim=1)
        x = self.main_layers[0](x)
        x = self.linear(x.view(-1,x.shape[1]))
        return x
