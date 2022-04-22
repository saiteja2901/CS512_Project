from pyexpat import model
from turtle import forward
from pyrsistent import inc
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierBlock(nn.Module):
    def __init__(self, input_channels) :
        super(FourierBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels = input_channels * 2, out_channels = input_channels *2, kernel_size = 1)
        # The size of the channels is doubled as the number of channels is doubled after real FFT layer.
        self.bn = nn.BatchNorm2d(input_channels * 2)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = torch.fft.rfftn(x, dim = (-2, -1), norm = "ortho")
        #return tensors in the shape (batch, c, h, w/2, 2). 2 is added to dimensions as fft outputs has real and imaginary parts.
        #The widht is halved as the output is conjugate complex symmetry or Hermitian symmetry for Real valued inputs.
        x = torch.stack((x.real, x.imag), dim = 4)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        #returns tensors in the shape(batch, c, 2, h, w/2)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        #returns tensors in the shape(batch, 2c, h, w/2)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = x.view(x.shape[0], -1, 2, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2).contiguous()
        #return tesnsor to shape (batch, c, 2, h, w/2) and rearrange to (batch, c, h, w/2, 2)
        x = torch.complex(x[..., 0], x[..., 1])
        #convert individual points of size 2 to imaginary points
        output = torch.fft.irfftn(x)#, s = x.shape[-2:], dim = (-2, -1), norm = "ortho")

        return output


class SpectralTransform(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SpectralTransform, self).__init__()
        #spectral transfrom computes the global convolution i.e., has gloabal receptive field
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = output_channels // 2, kernel_size = 1)
        self.bn = nn.BatchNorm2d(output_channels // 2)
        self.relu = nn.ReLU(inplace = True)
        self.fourier = FourierBlock(input_channels = output_channels // 2)
        self.conv2 = nn.Conv2d(in_channels = output_channels // 2, out_channels = output_channels, kernel_size = 1)#, padding = 1, padding_mode = "reflect")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x_rfft = self.fourier(x)
        output = self.conv2(x + x_rfft)

        return output


class LocalAndGlobal(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LocalAndGlobal, self).__init__()
        self.ratio_gout = 1
        in_cg = int(input_channels * self.ratio_gout)
        in_cl = input_channels - in_cg
        out_cg = int(output_channels * self.ratio_gout)
        out_cl = output_channels - out_cg
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_channels = in_cl, out_channels= out_cl, kernel_size = 3, padding = 1, padding_mode = "reflect")
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_channels = in_cl, out_channels= out_cg, kernel_size = 3, padding = 1, padding_mode = "reflect")
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_channels = in_cg, out_channels= out_cl, kernel_size = 3, padding = 1, padding_mode = "reflect")
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg)
    
    def forward(self, x):
        x_local, x_global = x if type(x) is tuple else (x, 0)
        output_local, output_global = 0, 0
        if self.ratio_gout != 1:
            output_local = self.convl2l(x_local) + self.convg2l(x_global)
        
        if self.ratio_gout != 0:
            if type(x) is tuple:
                output_global = self.convl2g(x_local) + self. convg2g(x_global)
            else:
                 output_global = self. convg2g(x_local)

        return output_local, output_global


class FFC(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FFC, self).__init__()
        self.ratio_gout = 1
        self.landg = LocalAndGlobal(input_channels, output_channels)
        lnorm = nn.Identity if self.ratio_gout == 1 else nn.BatchNorm2d
        gnorm = nn.Identity if self.ratio_gout == 0 else nn.BatchNorm2d
        global_channels = int(output_channels * self.ratio_gout)
        self.bn_local = lnorm(output_channels - global_channels)
        self.bn_global = gnorm(global_channels)

        lact = nn.Identity if self.ratio_gout == 1 else nn.ReLU
        gact = nn.Identity if self.ratio_gout == 0 else nn.ReLU
        self.relu_local = lact(inplace = True)
        self.relu_global = gact(inplace = True)

    def forward(self, x):
        x_local, x_global = self.landg(x)

        x_local = self.bn_local(x_local)
        x_local = self.relu_local(x_local)
        x_global = self.bn_global(x_global)
        x_global = self.relu_global(x_global)
        return x_local, x_global


class FCCResNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FCCResNet, self).__init__()
        self.ffc1 = FFC(input_channels, output_channels)
        self.ffc2 = FFC(input_channels, output_channels)
    
    def forward(self, x):
        x_local, x_global = x if type(x) is tuple else (x, 0)
        prev_local, prev_global = x_local, x_global
        x_local, x_global = self.ffc1(((x_local, x_global)))
        x_local, x_global = self.ffc2((x_local, x_global))
        # print("prev_local", prev_local.shape)
        # print("xlocal", x_local.shape)
        x_local = x_local + prev_local
        x_global = x_global + prev_global
        return x_local, x_global

class ConcatLocalAndGlobal(nn.Module):
    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        if not torch.is_tensor(x_g):
            return x_l
        if not torch.is_tensor(x_l):
            return x_g
        return torch.cat((x_l, x_g), dim = 1)


class GeneratorInpainting(nn.Module):
    def __init__(self, config, kind, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        super().__init__()

        model = [#nn.ReflectionPad2d(1),
                FFC(input_nc, 16),# 3 layers of downsampling
                FFC(16, 32),
                # FFC(64, 128),
                FCCResNet(32, 32),#9 layers of residual blocks
                FCCResNet(32, 32),
                FCCResNet(32, 32),
                #FCCResNet(128, 128),
                # FCCResNet(128, 128),
                # FCCResNet(128, 128),
                # FCCResNet(128, 128),
                # FCCResNet(128, 128),
                # FCCResNet(128, 128),
                ConcatLocalAndGlobal(),#3 layers of upscaling 
                #nn.ConvTranspose2d(128, 64, kernel_size = 3),
                nn.ConvTranspose2d(32, 16, kernel_size = 3),
                nn.Conv2d(16, output_nc, kernel_size = 3),
                nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
