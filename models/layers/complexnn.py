import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb

def _istuple(x):   return isinstance(x, tuple)
def _mktuple2d(x): return x if _istuple(x) else (x,x)

# Utility functions for initialization
def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
    if not fanin:
        fanin = 1
        for p in W1.shape[1:]:
            fanin *= p
    scale  = float(gain) / float(fanin)
    theta  = torch.empty_like(Wr).uniform_(-math.pi/2, +math.pi/2)
    rho    = np.random.rayleigh(scale, tuple(Wr.shape))
    rho    = torch.tensor(rho).to(Wr)
    Wr.data.copy_(rho * theta.cos())
    Wi.data.copy_(rho * theta.sin())

# Layers

def get_power_reciprocal(xr, xi, eps=1e-10):
    return 1/(xr*xr + xi*xi + eps)

def complex_left_inverse_by_real(xr, xi, eps=1e-8):
    # xr, xi: NxMxFxT --> NxTxFxM --> NTFxMx1 : use batch-wise inverse
    N, M, F, T = xr.size()
    xr = xr.transpose(1, 3).contiguous().view(N * T * F, M, 1) # NTFxMx1
    xi = xi.transpose(1, 3).contiguous().view(N * T * F, M, 1) # NTFxMx1
    xr_t = xr.transpose(1, 2)
    xi_t = xi.transpose(1, 2)

    cr = torch.reciprocal(torch.bmm(xr_t, xr)+eps) # for 1x1 matrix, inverse is reciprocal (special case)
    cr = torch.bmm(xi*cr, xr_t)
    cr = xr + torch.bmm(cr, xi)
    cr_t = cr.transpose(1, 2)

    ci = torch.reciprocal(torch.bmm(xi_t, xi)+eps) # for 1x1 matrix, inverse is reciprocal (special case)
    ci = torch.bmm(xr*ci, xi_t)
    ci = -(xi + torch.bmm(ci, xr))
    ci_t = ci.transpose(1, 2)

    # yr = torch.bmm(torch.inverse(torch.bmm(cr_t, cr)), cr_t) # left-inverse # NTFx1xM
    #pdb.set_trace()
    yr = torch.bmm(cr_t, cr)
    #yr = torch.inverse(yr)
    yr = torch.reciprocal(yr+eps) # special case
    #yr = torch.bmm(yr, cr_t)
    yr = yr*cr_t


    #yi = torch.bmm(torch.inverse(torch.bmm(ci_t, ci)), ci_t) # left-inverse # NTFx1xM
    yi = torch.bmm(ci_t, ci)
    yi = torch.reciprocal(yi+eps)
    yi = yi*ci_t

    yr = yr.view(N, T, F, M).transpose(1, 3)
    yi = yi.view(N, T, F, M).transpose(1, 3)

    return yr, yi




class ComplexConvWrapper(nn.Module):
    def __init__(self, conv_module, w_init_std, *args, **kwargs):
        super(ComplexConvWrapper, self).__init__()
        self.conv_re = conv_module(*args, **kwargs)
        self.conv_im = conv_module(*args, **kwargs)
        self.w_init_std = w_init_std

    def reset_parameters(self):
        if(self.w_init_std == 0):
            fanin = self.conv_re.in_channels // self.conv_re.groups
            for s in self.conv_re.kernel_size: fanin *= s
            complex_rayleigh_init(self.conv_re.weight, self.conv_im.weight, fanin)
        else:
            self.conv_re.weight.normal_(mean=0, stddev=self.w_init_std)
            self.conv_im.weight.normal_(mean=0, stddev=self.w_init_std)

        if self.conv_re.bias is not None:
            self.conv_re.bias.data.zero_()
            self.conv_im.bias.data.zero_()

    def forward(self, xr, xi):
        real = self.conv_re(xr) - self.conv_im(xi)
        imag = self.conv_re(xi) + self.conv_im(xr)
        return real, imag

# Real-valued network module for complex input
class RealConvWrapper(nn.Module):
    def __init__(self, conv_module, *args, **kwargs):
        super(RealConvWrapper,self).__init__()
        self.conv_re = conv_module(*args, **kwargs)

    #def forward(self, xr, xi): # chanil's
#        real = self.conv_re(xr)
#        imag = self.conv_re(xi)
#        return real, imag

    def forward(self, x):
        out = self.conv_re(x)

        return out



class CLeakyReLU(nn.LeakyReLU):
    def forward(self, xr, xi):
        return F.leaky_relu(xr, self.negative_slope, self.inplace),\
                F.leaky_relu(xi, self.negative_slope, self.inplace)

class ComplexDepthwiseConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,  bias=True):
        super(ComplexDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.groups = in_channels
        self.out_channels = self.groups*out_channels
        self.kernel_size = _mktuple2d(kernel_size)
        self.stride = _mktuple2d(stride)
        self.padding = _mktuple2d(padding)
        self.dilation = _mktuple2d(dilation)

        self.Wreal = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                    self.in_channels // self.groups,
                                                    *self.kernel_size))
        self.Wimag = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                    self.in_channels // self.groups,
                                                    *self.kernel_size))

        if bias:
            self.Breal = torch.nn.Parameter(torch.Tensor(self.out_channels))
            self.Bimag = torch.nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("Breal", None)
            self.register_parameter("Bimag", None)
        self.reset_parameters()

    def reset_parameters(self):
        fanin = self.in_channels // self.groups
        for s in self.kernel_size: fanin *= s
        complex_rayleigh_init(self.Wreal, self.Wimag, fanin)

        if self.Breal is not None and self.Bimag is not None:
            self.Breal.data.zero_()
            self.Bimag.data.zero_()

    def forward(self, xr, xi):
        #pdb.set_trace()
        yrr = torch.nn.functional.conv2d(xr, self.Wreal, self.Breal, self.stride, self.padding, self.dilation, self.groups)
        yri = torch.nn.functional.conv2d(xr, self.Wimag, self.Bimag, self.stride, self.padding, self.dilation, self.groups)
        yir = torch.nn.functional.conv2d(xi, self.Wreal, None, self.stride, self.padding, self.dilation, self.groups)
        yii = torch.nn.functional.conv2d(xi, self.Wimag, None, self.stride, self.padding, self.dilation, self.groups)

        Yr = yrr - yii
        Yi = yri + yir

        Yr = Yr.view(Yr.size(0), self.in_channels, -1, Yr.size(3))
        Yi = Yi.view(Yi.size(0), self.in_channels, -1, Yi.size(3))
        return Yr, Yi


class ComplexDepthwiseConvTransposed(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(ComplexDepthwiseConvTransposed, self).__init__()
        self.in_channels = in_channels
        self.groups = in_channels
        self.out_channels = self.groups * out_channels
        self.kernel_size = _mktuple2d(kernel_size)
        self.stride = _mktuple2d(stride)
        self.padding = _mktuple2d(padding)
        self.dilation = _mktuple2d(dilation)

        # ver 2:
        # #output padding must be smaller than either stride or dilation, but got adjH: 1 adjW: 1 dH: 1 dW: 1 dilationH: 1 dilationW: 1
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
        self.Wreal = torch.nn.Parameter(torch.Tensor(self.in_channels,
                                                        self.out_channels // self.groups,
                                                        *self.kernel_size))
        self.Wimag = torch.nn.Parameter(torch.Tensor(self.in_channels,
                                                        self.out_channels // self.groups,
                                                        *self.kernel_size))

        if bias:
            self.Breal = torch.nn.Parameter(torch.Tensor(self.out_channels))
            self.Bimag = torch.nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("Breal", None)
            self.register_parameter("Bimag", None)
        self.reset_parameters()

    def reset_parameters(self):
        fanin = self.in_channels // self.groups
        for s in self.kernel_size: fanin *= s
        complex_rayleigh_init(self.Wreal, self.Wimag, fanin)

        if self.Breal is not None and self.Bimag is not None:
            self.Breal.data.zero_()
            self.Bimag.data.zero_()

    def forward(self, xr, xi):
        #pdb.set_trace()
        output_padding = 0
        yrr = torch.nn.functional.conv_transpose2d(xr, self.Wreal, self.Breal, self.stride, self.padding, output_padding, self.groups, self.dilation)
        yri = torch.nn.functional.conv_transpose2d(xr, self.Wimag, self.Bimag, self.stride, self.padding, output_padding, self.groups, self.dilation)
        yir = torch.nn.functional.conv_transpose2d(xi, self.Wreal, None, self.stride, self.padding, output_padding, self.groups, self.dilation)
        yii = torch.nn.functional.conv_transpose2d(xi, self.Wimag, None, self.stride, self.padding, output_padding, self.groups, self.dilation)

        Yr = yrr - yii
        Yi = yri + yir

        Yr = Yr.view(Yr.size(0), self.in_channels, -1, Yr.size(3))
        Yi = Yi.view(Yi.size(0), self.in_channels, -1, Yi.size(3))
        return Yr, Yi

class transpose_channel_freq_layer(torch.nn.Module): # temporarily used
    def __init__(self, module):
        super(transpose_channel_freq_layer, self).__init__()
        self.module = module

    def forward(self, xr, xi):
        xr = xr.transpose(1, 2) # NxFxCxT --> NxCxFxT
        xi = xi.transpose(1, 2) # NxFxCxT --> NxCxFxT

        yr, yi= self.module(xr, xi)

        yr = yr.transpose(1, 2)  # NxCxFxT --> NxFxCxT
        yi = yi.transpose(1, 2)  # NxCxFxT --> NxFxCxT

        return yr, yi


# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch
class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True):
        super(ComplexBatchNorm, self).__init__()
        self.num_features        = num_features
        self.eps                 = eps
        self.momentum            = momentum
        self.affine              = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)
        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(num_features))
            self.register_buffer('RMi',  torch.zeros(num_features))
            self.register_buffer('RVrr', torch.ones (num_features))
            self.register_buffer('RVri', torch.zeros(num_features))
            self.register_buffer('RVii', torch.ones (num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, xr, xi):
        self._check_input_dim(xr, xi)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau   = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        return yr, yi

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)
