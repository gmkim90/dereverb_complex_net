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
    def __init__(self, conv_cfg, leaky_slope, use_depthwise=False, nFreq=0, use_bn=True):
        super(Encoder, self).__init__()
        if(use_depthwise):
            out_channels = conv_cfg[1]
            kernel_sz_ch = conv_cfg[0]
            kernel_sz_time = conv_cfg[2][1]
            stride_ch = 1
            stride_time = conv_cfg[3][1]
            padding_ch = 0
            padding_time = int((kernel_sz_time-1)/2)
            self.conv = dcnn.ComplexDepthwiseConv(in_channels=nFreq, out_channels=out_channels,
                                                  kernel_size = (kernel_sz_ch, kernel_sz_time),
                                                  stride = (stride_ch, stride_time),
                                                  padding = (padding_ch, padding_time), bias=False)
        else:
            self.conv = dcnn.ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.use_bn = use_bn
        if(use_bn):
            if(use_depthwise):
                self.bn = dcnn.transpose_channel_freq_layer(dcnn.ComplexBatchNorm(conv_cfg[1]))
            else:
                self.bn = dcnn.ComplexBatchNorm(conv_cfg[1])
        self.act = dcnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi):
        if(self.use_bn):
            xr, xi = self.act(*self.bn(*self.conv(xr, xi)))
        else:
            xr, xi = self.act(*self.conv(xr, xi))
        return xr, xi

class Decoder(nn.Module):
    def __init__(self, dconv_cfg, leaky_slope, use_depthwise=False, nFreq=0, use_bn=True):
        super(Decoder, self).__init__()
        if(use_depthwise):
            self.skip_dim = 2
            out_channels = dconv_cfg[1]
            kernel_sz_time = dconv_cfg[2][1]
            stride_ch = 1
            stride_time = dconv_cfg[3][1]

            # ver 1
            #kernel_sz_ch = dconv_cfg[0]
            #padding_ch = 0

            # ver 2
            kernel_sz_ch = 2
            padding_ch = int(dconv_cfg[0]/2)

            padding_time = int((kernel_sz_time-1)/2)
            self.dconv = dcnn.ComplexDepthwiseConvTransposed(in_channels=nFreq, out_channels=out_channels,
                                                  kernel_size = (kernel_sz_ch, kernel_sz_time),
                                                  stride = (stride_ch, stride_time),
                                                  padding = (padding_ch, padding_time), bias=False)

        else:
            self.dconv = dcnn.ComplexConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
            self.skip_dim = 1
        self.use_bn = use_bn
        if(use_bn):
            if(use_depthwise):
                self.bn = dcnn.transpose_channel_freq_layer(dcnn.ComplexBatchNorm(dconv_cfg[1]))
            else:
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

class Unet(nn.Module):
    def __init__(self, cfg, loss_type = 'cossim_time', nMic = 1, reverb_frame = 1, use_depthwise=False, nFreq=0, use_bn=True, input_type ='complex', ds_rate = 1, inverse_type='no', f_start_ratio=0, f_end_ratio=1,
                 ec_decomposition=False, carrier_input_indep=False, ec_bias=False, carrier_scale=0.001):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList()
        self.reverb_frame = reverb_frame
        self.input_type = input_type
        self.ds_rate = ds_rate
        self.use_depthwise = use_depthwise
        self.nMic = nMic
        #self.power_reciprocal_conjugate = power_reciprocal_conjugate
        self.inverse_type = inverse_type
        if(use_depthwise):
            self.skip_dim = 2
        else:
            self.skip_dim = 1

        self.ec_decomposition = ec_decomposition # envelope-carrier decomposition
        self.carrier_input_indep = carrier_input_indep
        self.ec_bias = ec_bias

        self.f_start_ratio = f_start_ratio
        self.f_end_ratio = f_end_ratio

        if(self.ec_decomposition):
            nFreq_orig = (nFreq - 1)*ds_rate + 1
            if(self.carrier_input_indep):
                self.carrier_freq = torch.FloatTensor(nMic, nFreq_orig, 3).uniform_(-carrier_scale, carrier_scale).cuda()
                self.carrier_bias = torch.FloatTensor(nMic, nFreq_orig).uniform_(-carrier_scale, carrier_scale).cuda()
                self.carrier_freq = nn.Parameter(self.carrier_freq)
                self.carrier_bias = nn.Parameter(self.carrier_bias)
            if(self.ec_bias):
                self.W_bias_r = torch.zeros(nMic, nFreq_orig).cuda()
                self.W_bias_i = torch.zeros(nMic, nFreq_orig).cuda()
                self.W_bias_r = nn.Parameter(self.W_bias_r)
                self.W_bias_i = nn.Parameter(self.W_bias_i)

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

        if(input_type == 'complex_ratio' or input_type == 'log_complex_ratio' or input_type == 'log_complex_ratio_unwrap' or input_type == 'log_complex_ratio_cos'):
            cfg['encoders'][0][0] = nMic-1
        else:
            cfg['encoders'][0][0] = nMic
        for conv_cfg in cfg['encoders']:
            self.encoders.append(Encoder(conv_cfg, cfg['leaky_slope'], use_depthwise=use_depthwise, nFreq=nFreq, use_bn=use_bn))

        cfg['decoders'][-1][1] = nMic*reverb_frame
        if(self.ec_decomposition and not self.carrier_input_indep):
            cfg['decoders'][-1][1] = cfg['decoders'][-1][1]*3 # original + (x,y,z,bias) = 5 --> 5 = 1 + 2*2 (real/imag), 3 = 1+2

        self.decoders = nn.ModuleList()
        for dconv_cfg in cfg['decoders'][:-1]:
            self.decoders.append(Decoder(dconv_cfg, cfg['leaky_slope'], use_depthwise=use_depthwise, nFreq=nFreq, use_bn=use_bn))

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
                self.decoders_additional.append(Decoder(cfg_additional_layer, cfg['leaky_slope'], use_bn=use_bn))

        # Last decoder doesn't use BN & LeakyReLU. Use bias.
        if(use_depthwise): # use conv instead of transposed conv avoid checkboard effect
            conv_cfg = cfg['decoders'][-1]
            out_channels = conv_cfg[1]
            kernel_sz_ch = conv_cfg[0]
            kernel_sz_time = conv_cfg[2][1]
            stride_ch = 1
            stride_time = conv_cfg[3][1]
            padding_ch = 0
            padding_time = int((kernel_sz_time-1)/2)
            self.last_decoder = dcnn.ComplexDepthwiseConv(in_channels=nFreq, out_channels=out_channels,
                                                  kernel_size = (kernel_sz_ch, kernel_sz_time),
                                                  stride = (stride_ch, stride_time),
                                                  padding = (padding_ch, padding_time), bias=True)
        else:
            self.last_decoder = dcnn.ComplexConvWrapper(nn.ConvTranspose2d, *cfg['decoders'][-1], bias=True)

        #pdb.set_trace()
        self.ratio_mask_type = cfg['ratio_mask']

        if(loss_type == 'cossim_mag' or loss_type == 'sInvSDR_mag'):
            self.output_phase = False
        else:
            self.output_phase = True

    def forward(self, xr, xi):
    #def forward(self, x_list):
        #xr, xi = x_list[0], x_list[1]
        input_real, input_imag = xr, xi

        #pdb.set_trace()

        if(self.use_depthwise):
            xr = xr.transpose(1, 2) # NxMxFxT --> NxFxMxT
            xi = xi.transpose(1, 2)  # NxMxFxT --> NxFxMxT
        skips = list()

        if(not self.input_type == 'complex'):
            xr, xi = self.complex_ratio(xr, xi)
            if(self.ds_rate > 1):
                xr, xi = self.downsample_freq(xr, xi)

            if(self.input_type == 'log_complex_ratio'): # downsample on complex_ratio domain & and then take log
                eps = 1e-8
                magnitude = torch.sqrt(xr*xr + xi*xi + eps)
                phase = torch.atan2(xi, xr)
                xr, xi = torch.log(magnitude), phase
            elif(self.input_type == 'log_complex_ratio_unwrap'): # downsample on complex_ratio domain & and then take log
                eps = 1e-8
                magnitude = torch.sqrt(xr*xr + xi*xi + eps)
                phase = torch.atan2(xi, xr)
                phase = torch.FloatTensor(np.unwrap(phase.cpu().numpy(), axis=2)).cuda()
                xr, xi = torch.log(magnitude), phase

        # sio.savemat('IMR.mat', {'IMR_real':xr.data.cpu().numpy(), 'IMR_imag':xi.data.cpu().numpy()})

        # slice xr, xi, input_real, input_imag by frequency
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
        xr, xi = self.last_decoder(xr, xi) # get mask

        if(self.use_depthwise): # transpose back
            xr = xr.transpose(1, 2) # NxFxMxT --> NxMxFxT
            xi = xi.transpose(1, 2) # NxFxMxT --> NxMxFxT

        if(self.reverb_frame > 1):
            # step 1) zero padding as much as reverb time
            input_real = Func.pad(input_real, (0, self.reverb_frame, 0, 0))  # (Ts,Te, Fs, Fe)
            input_imag = Func.pad(input_imag, (0, self.reverb_frame, 0, 0))  # (Ts,Te, Fs,Fe)

            # step 2) transpose
            input_real = input_real.transpose(0, 3).contiguous() # (NxMxFx(T+Lh)) --> ((T+Lh)xMxFxN)
            input_imag = input_imag.transpose(0, 3).contiguous() # (NxMxFx(T+Lh)) --> ((T+Lh)xMxFxN)

            # step 3) splicing
            input_real = self.reverb_splicing(input_real)
            input_imag = self.reverb_splicing(input_imag)

            # step 4) transpose back
            input_real = input_real.transpose(0, 3) # (Nx(Lh*M)xTxF) --> (Nx(Lh*M)xFxT) # ver 1
            input_imag = input_imag.transpose(0, 3) # (Nx(Lh*M)xTxF) --> (Nx(Lh*M)xFxT) # ver 1

        if(self.inverse_type == 'left'):
            xr, xi = dcnn.complex_left_inverse_by_real(xr, xi)

        # if(self.power_reciprocal_conjugate):
        elif(self.inverse_type == 'power_reciprocal_conjugate'):
            power_reciprocal = dcnn.get_power_reciprocal(xr, xi)
            power_reciprocal = power_reciprocal/self.nMic
            xr = xr*power_reciprocal
            xi = -xi*power_reciprocal

        if(self.ec_decomposition):
            src_pos = torch.FloatTensor(x_list[2]).cuda() # Nx3
            N, M, F, T = xr.size()
            if(self.carrier_input_indep):
                src_pos_expand = src_pos.unsqueeze(1).unsqueeze(1).expand(N, M, F, 3)
                carrier_freq_expand = self.carrier_freq.unsqueeze(0).expand(N, M, F, 3)
                carrier_bias_expand = self.carrier_bias.unsqueeze(0).expand(N, M, F)
                carrier_linear = torch.sum(src_pos_expand*carrier_freq_expand, dim=3) + carrier_bias_expand # NxMxF
                carrier_real = torch.cos(carrier_linear).unsqueeze(3).expand(N, M, F, T) # NxMxFxT
                carrier_imag = torch.sin(carrier_linear).unsqueeze(3).expand(N, M, F, T) # NxMxFxT

                env_r = xr
                env_i = xi
            else:
                # ver 1.
                M = int(M/3) # (3 = 1 + 2(real/imag)*2(=(x,y,z,bias))
                #pdb.set_trace()
                src_pos_expand = src_pos.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(N, M, F, T, 3)
                carrier_freq = torch.cat((xr[:, M:(2*M), :, :].unsqueeze(4), xr[:, (2*M):, :, :].unsqueeze(4)), dim=4)
                carrier_freq = torch.cat((carrier_freq, xi[:, M:(2*M), :, :].unsqueeze(4)), dim=4) # NxMxFxTx3
                carrier_bias = xi[:, (2*M):, :, :] # NxMxFxT
                carrier_linear = torch.sum(src_pos_expand*carrier_freq, dim=4) + carrier_bias # NxMxFxT
                carrier_real = torch.cos(carrier_linear) # NxMxFxT
                carrier_imag = torch.sin(carrier_linear) # NxMxFxT

                env_r = xr[:, :M, :, :] # envelope
                env_i = xi[:, :M, :, :] # envelope

            # Envelope*Carrier
            xr = env_r*carrier_real - env_i*carrier_imag
            xi = env_r*carrier_imag + env_i*carrier_real

            # Add optional learnable bias
            if(self.ec_bias):
                W_bias_r_expand = self.W_bias_r.unsqueeze(0).unsqueeze(3).expand(N, M, F, T) # MxF --> NxMxFxT
                W_bias_i_expand = self.W_bias_i.unsqueeze(0).unsqueeze(3).expand(N, M, F, T) # MxF --> NxMxFxT

                xr = xr + W_bias_r_expand
                xi = xi + W_bias_i_expand

        ratio_mask_fn = self.get_ratio_mask(xr, xi)


        if(self.ec_decomposition and not self.training):
            #return [ratio_mask_fn(input_real, input_imag), env_r, env_i, carrier_real, carrier_imag, self.W_bias_r, self.W_bias_i] # suspicious memory leak
            return ratio_mask_fn(input_real, input_imag), env_r, env_i, carrier_real, carrier_imag, self.W_bias_r, self.W_bias_i

        else:
            #return [ratio_mask_fn(input_real, input_imag)] # suspicious memory leak
            return ratio_mask_fn(input_real, input_imag)


    def get_ratio_mask(self, outr, outi):
        def inner_fn(r, i):
            if self.ratio_mask_type == 'realimag':
                #pdb.set_trace()
                #return [torch.sum(outr*r - outi*i, dim=1), torch.sum(outr*i + outr*i, dim=1)], [outr, outi] # suspicious memory leak
                return torch.sum(outr * r - outi * i, dim=1), torch.sum(outr * i + outr * i, dim=1), outr, outi

            elif self.ratio_mask_type == 'pow_only':
                mask_pow = outr*outr + outi*outi
                input_pow = r*r + i*i
                output_pow = torch.sum(mask_pow*input_pow, dim=1)
                return [output_pow], [mask_pow]

            elif self.ratio_mask_type == 'magphs':
                mag_mask = torch.sqrt(outr**2 + outi**2)
                phase_rotate = torch.atan2(outi, outr)

                mag = mag_mask * torch.sqrt(r**2 + i**2)
                phase = phase_rotate + torch.atan2(i, r)

                #return mag * torch.cos(phase), mag * torch.sin(phase) # single-mic
                return torch.sum(mag * torch.cos(phase), dim=1), torch.sum(mag * torch.sin(phase), dim=1), mag_mask, phase_rotate # multi-mic

            elif self.ratio_mask_type == 'no_mask':
                sr = torch.sum(outr, dim=1)
                si = torch.sum(outi, dim=1)
                return sr, si, outr, outi


        return inner_fn

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

    def reverb_splicing(self, x):
        # x: NxMxFx(T+Lh) # wo/ transpose
        #N, M, F, T = x.size()

        # x: NxMx(T+Lh)xF # w/ transpose (ver 1)
        #N, M, T, F = x.size()

        # x: NxFxMx(T+Lh) (ver 2)
        # N, F, M, T = x.size()
        #out = torch.FloatTensor(N, F, M * self.reverb_frame, T - self.reverb_frame)

        # x: (T+Lh)xNxFxM (ver 3)
        T, M, F, N = x.size()
        out = torch.FloatTensor(T - self.reverb_frame, M * self.reverb_frame, F, N).cuda()


        #x[:, :, :, t:t+Lh] # NxMxFxLh # wo/ transpose
        #x[:, :, t:t+Lh, :].view(N, -1,  F) # Nx(M*Lh)xF

        for t in range(T-self.reverb_frame):
            #out[:, :, :, t] = x[:, :, t:t+self.reverb_frame, :].view(N, -1, F) # ver 1
            #out[:, :, :, t] = x[:, :, :, t:t + self.reverb_frame].view(N, F, -1) # ver 2
            out[t, :, :, :] = x[t:t + self.reverb_frame, :, :, : ].view(-1, F, N) # ver 3

        return out

    def complex_ratio(self, xr, xi):
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

    def log_complex_ratio(self, xr, xi, unwrap = False, apply_cos=False):
        # xr, xr: NxMxFxT
        PI = math.pi
        TWOPI = PI * 2

        eps = 1e-10
        N, M, F, T = xr.size()
        yr, yi = torch.FloatTensor(N, M-1, F, T).cuda(), torch.FloatTensor(N, M-1, F, T).cuda()
        mic_magnitude = torch.sqrt(xr*xr + xi*xi + eps)
        mic_phase = torch.atan2(xi, xr)
        for m in range(M-1):
            yr[:, m] = torch.log(mic_magnitude[:, m+1] + eps) - torch.log(mic_magnitude[:, 0] + eps)
            yi[:, m] = mic_phase[:, m+1] - mic_phase[:, 0]

        yi = torch.remainder(yi + PI, TWOPI) - PI

        if(unwrap):
            yi = torch.FloatTensor(np.unwrap(yi.cpu().numpy(), axis=2)).cuda()

        elif(apply_cos):
            yi = torch.cos(yi)

        return yr, yi


class RealUnet(nn.Module):
    def __init__(self, cfg, loss_type = 'cossim_time', nMic = 1, use_bn=True):
        super(RealUnet, self).__init__()
        self.encoders = nn.ModuleList()
        self.nMic = nMic

        cfg['encoders'][0][0] = nMic*2 # *2 for real/imag concat
        for conv_cfg in cfg['encoders']:
            self.encoders.append(Encoder_real(conv_cfg, cfg['leaky_slope'], use_bn=use_bn))

        cfg['decoders'][-1][1] = nMic*2 # *2 for real/imag concat
        self.decoders = nn.ModuleList()
        for dconv_cfg in cfg['decoders'][:-1]:
            self.decoders.append(Decoder_real(dconv_cfg, cfg['leaky_slope'], use_bn=use_bn))

        self.last_decoder = dcnn.RealConvWrapper(nn.ConvTranspose2d, *cfg['decoders'][-1], bias=True)

        self.ratio_mask_type = cfg['ratio_mask']

        if(loss_type == 'cossim_mag' or loss_type == 'sInvSDR_mag'):
            self.output_phase = False
        else:
            self.output_phase = True

    def forward(self, xr, xi):
        input_real, input_imag = xr, xi

        input = torch.cat((input_real, input_imag), dim=1)
        x = input

        skips = list()
        # slice xr, xi, input_real, input_imag by frequency
        for n in range(len(self.encoders)-1):
            encoder = self.encoders[n]
            x = encoder(x)
            skips.append(x)

        encoder = self.encoders[len(self.encoders)-1]
        x = encoder(x)

        skip = None # First decoder input x is same as skip, drop skip.
        for decoder in self.decoders:
            x = decoder(x, skip)
            skip = skips.pop()

        #xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        x = cat_nopad(x, skip, dim=1)
        x = self.last_decoder(x) # get mask

        ratio_mask_fn = self.get_ratio_mask(x)
        #return ratio_mask_fn(input)
        return ratio_mask_fn(input_real, input_imag)

    def get_ratio_mask(self, out):
        out_r, out_i = torch.split(out, int(out.size(1)/2), dim=1)
        def inner_fn(mic_r, mic_i):
            #return [torch.sum(mic*out, dim=1)], [out]
            return [torch.sum(mic_r*out_r, dim=1), torch.sum(mic_i*out_i, dim=1)], [out_r, out_i]

        return inner_fn


class Encoder_real(nn.Module):
    def __init__(self, conv_cfg, leaky_slope, use_bn=True):
        super(Encoder_real, self).__init__()
        self.conv = dcnn.RealConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.use_bn = use_bn
        if(use_bn):
            self.bn = nn.BatchNorm2d(conv_cfg[1])
        self.act = nn.LeakyReLU(leaky_slope, inplace=True)

    def forward(self, x):
        if(self.use_bn):
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
            #x = self.act(*self.bn(*self.conv(x)))
        else:
            x = self.act(*self.conv(x))
        return x

class Decoder_real(nn.Module):
    def __init__(self, dconv_cfg, leaky_slope, use_bn=True):
        super(Decoder_real, self).__init__()
        self.dconv = dcnn.RealConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
        self.skip_dim = 1
        self.use_bn = use_bn

        if(use_bn):
            self.bn = nn.BatchNorm2d(dconv_cfg[1])
        self.act = nn.LeakyReLU(leaky_slope, inplace=True)

    def forward(self, x, skip=None):
        if skip is not None:
            x = cat_nopad(x, skip, dim=1)
        if(self.use_bn):
            #x = self.act(*self.bn(*self.dconv(x)))
            x = self.dconv(x)
            x = self.bn(x)
            x = self.act(x)
        else:
            x = self.act(*self.dconv(x))
        return x