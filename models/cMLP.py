import torch
import torch.nn as nn
import torch.nn.functional as Func
import models.layers.complexnn as dcnn
import numpy as np
import math
import scipy.io as sio

import pdb

def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
    if not fanin:
        fanin = 1
        for p in W1.shape[1:]:
            fanin *= p

    scale = float(gain)/float(fanin)
    theta  = torch.empty_like(Wr).uniform_(-math.pi/2, +math.pi/2)
    #theta = torch.zeros(Wr.size()).uniform_(-math.pi / 2, +math.pi / 2)
    rho = np.random.rayleigh(scale, tuple(Wr.shape))
    rho = torch.tensor(rho).to(Wr)
    Wr.data.copy_(rho*theta.cos())
    Wi.data.copy_(rho*theta.sin())

class cLinear(nn.Module):
    def __init__(self, nIn = 0, nOut = 256, bias=True):
        super(cLinear, self).__init__()
        self.nIn = nIn
        self.nOut = nOut

        self.Wr = torch.nn.Parameter(torch.Tensor(nOut, nIn))
        self.Wi = torch.nn.Parameter(torch.Tensor(nOut, nIn))
        #self.Wr = torch.nn.Parameter(torch.Tensor(nIn, nOut))
        #self.Wi = torch.nn.Parameter(torch.Tensor(nIn, nOut))

        if bias:
            self.Br = torch.nn.Parameter(torch.Tensor(nOut))
            self.Bi = torch.nn.Parameter(torch.Tensor(nOut))
        else:
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        self.reset_parameters()


    def reset_parameters(self):
        complex_rayleigh_init(self.Wr, self.Wi, self.nIn)

        if self.Br is not None and self.Bi is not None:
            self.Br.data.zero_()
            self.Bi.data.zero_()

    def forward(self, xr, xi):
        #pdb.set_trace()
        yrr = Func.linear(xr, self.Wr, self.Br)
        yri = Func.linear(xr, self.Wi, self.Bi)
        yir = Func.linear(xi, self.Wr, None)
        yii = Func.linear(xi, self.Wi, None)

        return yrr - yii, yri + yir


class cLinear_Bn_Act(nn.Module):
    def __init__(self, nIn = 0, nOut = 256, bias=True, use_bn=True, use_act=True, leaky_slope=0.01):
        super(cLinear_Bn_Act, self).__init__()
        self.nIn = nIn
        self.nOut = nOut

        self.cLinear = cLinear(nIn = nIn, nOut = nOut, bias = bias)

        self.use_bn = use_bn
        self.use_act = use_act

        if(use_bn):
            self.bn = dcnn.ComplexBatchNorm(nOut)

        if(use_act):
            self.act = dcnn.CLeakyReLU(leaky_slope, inplace = True)


    def forward(self, xr, xi):
        if(self.use_bn and self.use_act):
            yr, yi = self.act(*self.bn(*self.cLinear(xr, xi)))
        else:
            yr, yi = self.cLinear(xr, xi)

        return yr, yi

class cMLP(nn.Module):
    def __init__(self, nLayer=1, nHidden=256, nMic = 1, nFreq=0, ds_rate = 1):
        super(cMLP, self).__init__()
        self.nFreq=nFreq
        self.nLayer = nLayer
        self.nMic = nMic
        assert(self.nMic == 2), 'currently, nMic == 2 is supported'
        self.nFreq = nFreq
        self.ds_rate = ds_rate

        self.net = nn.ModuleList()
        nInput = int((nFreq-1)/ds_rate + 1)
        for l in range(nLayer-1):
            #net.append(cLinear(nInput, nHidden, bias=False)) # with batchnorm, bias is unnecessary
            #net.append(nn.BatchNorm1d(num_features=nHidden))
            #net.append(nn.LeakyReLU(negative_slope=0.01))

            self.net.append(cLinear_Bn_Act(nIn = nInput, nOut = nHidden, bias=False, use_bn = True, use_act = True, leaky_slope = 0.01))
            nInput = nHidden

        #net.append(cLinear_Bn_Act(nIn = nHidden, nOut = nFreq, bias=True, use_bn = False, use_act = False))
        self.net.append(cLinear(nIn = nHidden, nOut = nFreq*self.nMic, bias=True))

        #self.net = nn.Sequential(*net) # not suitable for input with two arguments (e.g., xr, xi)

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

    def forward(self, xr, xi):
        xr, xi = self.complex_ratio(xr, xi) # Nx2xFx1 --> Nx1xFx1
        if (self.ds_rate > 1):
            xr, xi = self.downsample_freq(xr, xi)

        #pdb.set_trace()
        # valid for M=2, T=1
        xr = xr.squeeze() # Nx1xFx1 --> NxF
        xi = xi.squeeze() # Nx1xFx1 --> NxF

        # forward net
        #pdb.set_trace()
        #wr, wi = self.net(xr, xi) # NxF

        for n in range(self.nLayer):
            #pdb.set_trace()
            xr, xi = self.net[n](xr, xi)

        return xr.view(xr.size(0), self.nMic, self.nFreq).unsqueeze(3), xi.view(xi.size(0), self.nMic, self.nFreq).unsqueeze(3)

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

