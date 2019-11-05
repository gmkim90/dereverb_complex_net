import torch
import torch.nn as nn
import torch.nn.functional as Func
import models.layers.complexnn as dcnn
import numpy as np
import math
import scipy.io as sio

import pdb

# NOTE: Use Complex Ops for DCUnet when implemented
# Reference:
#  > Progress: https://github.com/pytorch/pytorch/issues/755

def cat_nopad(x1, x2, dim):
    x1 = torch.cat([x1, x2], dim=dim)
    return x1

class Encoder(nn.Module):
    def __init__(self, conv_cfg, leaky_slope, use_bn=True, w_init_std=0):
        super(Encoder, self).__init__()

        # if w_init_std == 0, use rayleigh init

        self.conv = dcnn.ComplexConvWrapper(nn.Conv2d, w_init_std, *conv_cfg, bias=False)
        self.use_bn = use_bn
        if(use_bn):
            self.bn = dcnn.ComplexBatchNorm(conv_cfg[1])
        self.act = dcnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi):
        if(self.use_bn):
            xr, xi = self.act(*self.bn(*self.conv(xr, xi)))
        else:
            xr, xi = self.act(*self.conv(xr, xi))
        return xr, xi

class Decoder(nn.Module):
    def __init__(self, dconv_cfg, leaky_slope, use_bn=True, w_init_std=0):
        super(Decoder, self).__init__()

        # if w_init_std == 0, use rayleigh init

        self.dconv = dcnn.ComplexConvWrapper(nn.ConvTranspose2d, w_init_std, *dconv_cfg, bias=False)
        self.skip_dim = 1
        self.use_bn = use_bn
        if(use_bn):
            self.bn = dcnn.ComplexBatchNorm(dconv_cfg[1])
        self.act = dcnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi, skip=None):
        if skip is not None:
            #xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1) # no padding during forward
            #pdb.set_trace()
            xr, xi = cat_nopad(xr, skip[0], dim=self.skip_dim), cat_nopad(xi, skip[1], dim=self.skip_dim) # no padding during forward
        if(self.use_bn):
            xr, xi = self.act(*self.bn(*self.dconv(xr, xi)))
        else:
            xr, xi = self.act(*self.dconv(xr, xi))
        return xr, xi

class TDNN(nn.Module):
    def __init__(self, nLayer=1, nHidden=256, nMic = 1, nFreq=0):
        super(TDNN, self).__init__()
        #self.input_type = input_type
        self.nFreq=nFreq
        self.nMic = nMic

        net = []
        in_channel = nFreq*2
        for l in range(nLayer-1):
            net.append(nn.Conv1d(in_channels=in_channel, out_channels=nHidden, kernel_size=1))
            net.append(nn.BatchNorm1d(num_features=nHidden))
            net.append(nn.LeakyReLU(negative_slope=0.01))
            in_channel = nHidden

        net.append(nn.Conv1d(in_channels=nHidden, out_channels=nFreq*4, kernel_size=1))

        self.net = nn.Sequential(*net)

    def complex_ratio(self, xr, xi):
        #pdb.set_trace()
        # xr, xr: NxMxFxT
        #eps = 1e-10
        eps = 1e-16
        N, M, F, T = xr.size()
        yr, yi = torch.FloatTensor(N, M-1, F, T).cuda(), torch.FloatTensor(N, M-1, F, T).cuda()
        ref_pow = (xr[:, 0]*xr[:, 0] + xi[:, 0]*xi[:, 0] + eps)
        for m in range(M-1):
            yr[:, m] = (xr[:, m+1]*xr[:, 0] + xi[:, m+1]*xi[:, 0])/ref_pow
            yi[:, m] = (xi[:, m+1]*xr[:, 0] - xr[:, m+1]*xi[:, 0])/ref_pow

        return yr, yi

    def forward(self, xr, xi):
        input_real, input_imag = xr, xi # (Nx2xFxT), (Nx2xFxT)
        N, M, F, T = input_real.size()

        xr, xi = self.complex_ratio(xr, xi) # Nx(M-1)xFxT

        # valid for M=2
        xr = xr.squeeze(1)
        xi = xi.squeeze(1)

        # concat real & imag on freq dimension
        xri = torch.cat((xr, xi), dim=1) # Nx2FxT

        # forward net
        wri = self.net(xri) # Nx4FxT

        # divide
        wri_s = torch.split(wri, 2*F, dim=1) # (Nx2FxT), (Nx2FxT)
        wr = wri_s[0]
        wi = wri_s[1]
        wr = wr.view(N, M, F, T)
        wi = wi.view(N, M, F, T)

        # get output (complex arithmetic in here)
        sr = torch.sum(wr*input_real - wi*input_imag, dim=1)
        si = torch.sum(wr*input_imag + wi*input_real, dim=1)

        return sr, si, wr, wi


class Unet(nn.Module):
    def __init__(self, cfg, nMic = 1, input_type ='complex', ds_rate = 1, w_init_std=0, out_type = 'W'):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList()
        self.input_type = input_type
        self.ds_rate = ds_rate
        self.nMic = nMic
        self.w_init_std = w_init_std # if this is 0, use complex rayleigh init
        self.skip_dim = 1

        if(self.ds_rate > 1): # require downsample of mic ratio
            assert(math.log2(self.ds_rate) % 1 == 0)

            # make triangular filter
            filter_len = self.ds_rate*2-1
            self.ds_weight = torch.zeros(filter_len)
            self.ds_weight[:self.ds_rate] = torch.linspace(1/self.ds_rate, 1, steps=self.ds_rate)
            self.ds_weight[self.ds_rate:] = torch.linspace(1 - 1 / self.ds_rate, 1 / self.ds_rate, steps=self.ds_rate-1)

            # normalization
            self.ds_weight = self.ds_weight/self.ds_weight.sum()

            # expand
            self.ds_weight = self.ds_weight.view(1, 1, filter_len, 1).expand(nMic-1, nMic-1, filter_len, 1)

            # cuda
            self.ds_weight = self.ds_weight.cuda()

        if(input_type.find('ratio') >= 0):
            cfg['encoders'][0][0] = nMic-1
        else:
            cfg['encoders'][0][0] = nMic
        for conv_cfg in cfg['encoders']:
            self.encoders.append(Encoder(conv_cfg, cfg['leaky_slope']))

        if(out_type == 'W'):
            cfg['decoders'][-1][1] = nMic
        elif(out_type == 'S'):
            cfg['decoders'][-1][1] = 1

        self.decoders = nn.ModuleList()
        for dconv_cfg in cfg['decoders'][:-1]:
            self.decoders.append(Decoder(dconv_cfg, cfg['leaky_slope']))

        if(self.ds_rate > 1):
            self.decoders_additional = nn.ModuleList()

            # define additional decoder layer (without skip connection)
            nCH_in = cfg['decoders'][-1][0]
            nCH_out = nCH_in
            ksz_freq = 5
            ksz_time = 3
            upsample_freq = 2 # upsampling
            upsample_time = 1 # no upsampling
            pad_freq = int((ksz_freq-1)/2)
            pad_time = int((ksz_time-1)/2)

            cfg_additional_layer = [nCH_in, nCH_out, [ksz_freq, ksz_time], [upsample_freq, upsample_time], [pad_freq, pad_time] ]# [32, 32, [5, 3], [2, 1], [2, 1]]
            for r in range(int(math.log2(self.ds_rate))):
                self.decoders_additional.append(Decoder(cfg_additional_layer, cfg['leaky_slope']))

        self.last_decoder = dcnn.ComplexConvWrapper(nn.ConvTranspose2d, self.w_init_std, *cfg['decoders'][-1], bias=True)


    def forward(self, xr, xi, return_IMR=False):
        skips = list()
        if(return_IMR):
            input_real, input_imag = xr, xi

        if(not self.input_type == 'complex'):
            xr, xi = self.complex_ratio(xr, xi)
            if(self.ds_rate > 1):
                xr, xi = self.downsample_freq(xr, xi)

        for n in range(len(self.encoders)-1):
            encoder = self.encoders[n]
            xr, xi = encoder(xr, xi)
            skips.append((xr, xi))

        encoder = self.encoders[len(self.encoders)-1]
        xr, xi = encoder(xr, xi)

        skip = None # First decoder input x is same as skip, drop skip.
        for decoder in self.decoders:
            xr, xi = decoder(xr, xi, skip)
            skip = skips.pop()

        xr, xi = cat_nopad(xr, skip[0], dim=self.skip_dim), cat_nopad(xi, skip[1], dim=self.skip_dim) # no padding during forward

        if(self.ds_rate > 1):
            for decoder in self.decoders_additional:
                xr, xi = decoder(xr, xi, skip=None)
        out_real, out_imag = self.last_decoder(xr, xi) # output could be demixing weight or source (depending on loss function)

        if(not return_IMR):
            return torch.squeeze(out_real, dim=1), torch.squeeze(out_imag, dim=1)
        else: # only for analysis (i.e., save_activation=True)
            fr, fi = self.complex_ratio(input_real, input_imag)
            if (self.ds_rate > 1):
                fr, fi = self.downsample_freq(fr, fi)

            return torch.squeeze(out_real, dim=1), torch.squeeze(out_imag, dim=1), fr, fi

    def downsample_freq(self, xr, xi):
        # x : Nx(M-1)xFxT (x = Intermic ratio feature)
        F = xr.size(2)

        # case 1) intermediate frequencies
        xr_intermediate = xr[:, :, 1:-1, :] # exclude first & last frequency
        xr_intermediate_ds = Func.conv2d(xr_intermediate, self.ds_weight, bias=None, stride=(self.ds_rate, 1), padding=0, dilation=1)
        xi_intermediate = xi[:, :, 1:-1, :]  # exclude first & last frequency
        xi_intermediate_ds = Func.conv2d(xi_intermediate, self.ds_weight, bias=None, stride=(self.ds_rate, 1), padding=0, dilation=1)

        # case 2) first frequency
        # twosided spectrogram can be recovered from onesided spectrogram
        # caution about sign of imaginary
        # note: X[N-k] = conjugate(X[k]) (conjugate symmetric property)
        # reverse the sign of imaginary

        xr_first = torch.cat((xr[:, :, 1:self.ds_rate, :].flip(2), xr[:, :, 0:self.ds_rate, :]), dim=2)
        xr_first_ds = Func.conv2d(xr_first, self.ds_weight, bias=None, stride=(self.ds_rate, 1), padding = 0, dilation=1)
        xi_first = torch.cat((-xi[:, :, 1:self.ds_rate, :].flip(2), xi[:, :, 0:self.ds_rate, :]), dim=2) # reverse the first elements sign
        xi_first_ds = Func.conv2d(xi_first, self.ds_weight, bias=None, stride=(self.ds_rate, 1), padding=0, dilation=1)

        # case 3) last frequency
        xr_last = torch.cat((xr[:, :, F-self.ds_rate:F, :], xr[:, :, F-self.ds_rate:F-1, :].flip(2)), dim=2)
        xr_last_ds = Func.conv2d(xr_last, self.ds_weight, bias=None, stride=(self.ds_rate, 1), padding = 0, dilation=1)
        xi_last = torch.cat((xi[:, :, F-self.ds_rate:F, :], -xi[:, :, F-self.ds_rate:F-1, :].flip(2)), dim=2) # reverse the second elements sign
        xi_last_ds = Func.conv2d(xi_last, self.ds_weight, bias=None, stride=(self.ds_rate, 1), padding=0, dilation=1)

        # Concat all the components
        xr_ds = torch.cat((xr_first_ds, xr_intermediate_ds, xr_last_ds), dim=2)
        xi_ds = torch.cat((xi_first_ds, xi_intermediate_ds, xi_last_ds), dim=2)

        return xr_ds, xi_ds

    def complex_ratio(self, xr, xi):
        #pdb.set_trace()
        # xr, xr: NxMxFxT
        #eps = 1e-10
        eps = 1e-16
        N, M, F, T = xr.size()
        yr, yi = torch.FloatTensor(N, M-1, F, T).cuda(), torch.FloatTensor(N, M-1, F, T).cuda()
        ref_pow = (xr[:, 0]*xr[:, 0] + xi[:, 0]*xi[:, 0] + eps)
        for m in range(M-1):
            yr[:, m] = (xr[:, m+1]*xr[:, 0] + xi[:, m+1]*xi[:, 0])/ref_pow
            yi[:, m] = (xi[:, m+1]*xr[:, 0] - xr[:, m+1]*xi[:, 0])/ref_pow

        return yr, yi