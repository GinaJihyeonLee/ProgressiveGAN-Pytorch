import torch
import torch.nn as nn

def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, pixel=False, only=False, equalized=True):
  if equalized:
    layers.append(EqualizedConv2d(c_in, c_out, k_size, stride, pad))
  else:
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
  if not only:
    layers.append(nn.LeakyReLU(0.2))
    if bn: layers.append(nn.BatchNorm2d(c_out))
    elif pixel: layers.append(PixelNorm())
  return layers

def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, only=False, equalized=True):
  if equalized:
    layers.append(EqualizedConv2d(c_in, c_out, k_size, stride, pad))
  else:
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
  if not only:
    layers.append(nn.LeakyReLU(0.2))
    if bn: layers.append(nn.BatchNorm2d(c_out))
  return layers

def equalized_lr(x):
  return x

class PixelNorm(nn.Module):
  #pixel norm, instance norm?
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5

class EqualizedConv2d(nn.Module):
  #modify
  def __init__(self, c_in, c_out, k_size, stride, pad):
    super(EqualizedConv2d, self).__init__()
    conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
    nn.init.kaiming_normal_(conv.weight)
    self.conv =equalized_lr(conv)

  def forward(self, x):
    out = self.conv(x)
    return out

class EqualizedLinear(nn.Module):
  #modify
  def __init__(self, c_in, c_out):
    super(EqualizedLinear, self).__init__()
    linear = nn.Linear(c_in, c_out, bias=False)
    nn.init.kaiming_normal_(linear.weight)
    self.linear =equalized_lr(linear)

  def forward(self, x):
    out = self.linear(x)
    return out

# class StdConcat(nn.Module):

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
    layers = deconv(layers, self.config.nz, ndim, 4, 1, 3, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
    layers = deconv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
    return nn.Sequential(*layers)

  def init_rgb(self):
    layers=[]
    layers= deconv(layers,self.config.ngf,self.config.nc,1,1,0,only=True)
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
    layers= deconv(layers,c_in,self.config.nc,1,1,0,only=True)
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
      layers = deconv(layers, ndim*2, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
      layers = deconv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
    else:
      layers = deconv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
      layers = deconv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn, self.config.flag_pixel)
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
    self.linear=nn.Linear(config.ndf,1)

  def init_layers(self):
    layers=[]
    ndim = self.config.ndf
    #minibatch stddev
    layers = conv(layers, ndim, ndim, 3, 1, 1, self.config.flag_leaky, self.config.flag_bn)
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
    #upsampling? avgpool2d?
    layers.append(nn.AvgPool2d(kernel_size=2))
    return nn.Sequential(*layers), ndim

  def stage_up(self, resl):
    assert 3<=resl<=10
    print(f'growing network {pow(2,resl-1)}x{pow(2,resl-1)} to {pow(2,resl)}x{pow(2,resl)}')
    layers, ndim = self.middle_layers(resl)
    self.from_rgb_layers.append(self.from_rgb(ndim))
    self.main_layers.append(layers)

  def forward(self, x, alpha):
    downsample = nn.AvgPool2d(kernel_size=2)
    if alpha!=1:
      prev_x = self.from_rgb_layers[-2](downsample(x))
    x = self.from_rgb_layers[-1](x)
    x = self.main_layers[-1](x)
    if alpha!=1:
      x = alpha*x + (1-alpha)*prev_x
    for layer in reversed(self.main_layers[:-1]):
      x = layer(x)
    x = self.linear(x.view(-1,x.shape[1]))
    return x
