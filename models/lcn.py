import torch
import torch.nn as nn
import torch.nn.functional as Func
import models.layers.complexnn as dcnn
import math

import pdb

def cat_nopad(x1, x2, dim):
    x1 = torch.cat([x1, x2], dim=dim)
    return x1

class LCN(nn.Module):
    def __init__(self, nMic = 1, nFreq=0, nHidden=64, ksz_time = 3, nLayer = 3, use_bn=True, input_type ='complex', ds_rate = 1, reverb_frame=1, CW_freq=0, inverse_type='no'):
        super(LCN, self).__init__()
        self.input_type = input_type
        self.ds_rate = ds_rate
        self.reverb_frame = reverb_frame
        self.CW_freq = CW_freq
        self.nMic = nMic
        #self.power_reciprocal_conjugate = power_reciprocal_conjugate
        self.inverse_type =inverse_type

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

        if(input_type == 'complex_ratio' or input_type == 'log_complex_ratio' or input_type == 'log_complex_ratio_cos' or
                input_type == 'IMF_given_td_direct_only' or input_type == 'IMF_given_td_unwrap'):
            input_CH = nMic-1
        else:
            input_CH = nMic
        input_CH = input_CH*(2*self.CW_freq+1)

        self.lcn = nn.ModuleList()
        if (ds_rate == 1):
            for l in range(nLayer-1):
                conv_layer = dcnn.ComplexDepthwiseConv(in_channels=nFreq, out_channels=nHidden,
                                          kernel_size=(input_CH, ksz_time),
                                          stride=1, padding=(0, int((ksz_time-1)/2)), bias=False)
                self.lcn.append(conv_layer)

                if(use_bn):
                    bn = dcnn.transpose_channel_freq_layer(dcnn.ComplexBatchNorm(nHidden))
                    self.lcn.append(bn)

                act = dcnn.CLeakyReLU(0.01, inplace=True)
                self.lcn.append(act)

                input_CH = nHidden

            conv_layer = dcnn.ComplexDepthwiseConv(in_channels=nFreq, out_channels=nMic,
                                        kernel_size=(nHidden, ksz_time),
                                        stride=1, padding=(0, int((ksz_time-1)/2)), bias=True)
            self.lcn.append(conv_layer)

        elif(ds_rate > 1):
            for l in range(nLayer):
                conv_layer = dcnn.ComplexDepthwiseConv(in_channels=nFreq, out_channels=nHidden,
                                          kernel_size=(input_CH, ksz_time),
                                          stride=1, padding=(0, int((ksz_time-1)/2)), bias=False)
                self.lcn.append(conv_layer)

                if(use_bn):
                    bn = dcnn.transpose_channel_freq_layer(dcnn.ComplexBatchNorm(nHidden))
                    self.lcn.append(bn)

                act = dcnn.CLeakyReLU(0.01, inplace=True)
                self.lcn.append(act)

                input_CH = nHidden

            self.decoders_additional = nn.ModuleList()
            # define additional decoder layer (without skip connection)
            nCH_in = nHidden
            nCH_out = nCH_in
            ksz_freq = 5
            upsample_freq = 2 # upsampling
            upsample_time = 1 # no upsampling
            pad_freq = int((ksz_freq-1)/2)
            pad_time = int((ksz_time-1)/2)

            cfg_additional_layer = [nCH_in, nCH_out, [ksz_freq, ksz_time], [upsample_freq, upsample_time], [pad_freq, pad_time] ]# [32, 32, [5, 3], [2, 1], [2, 1]]
            cfg_additional_last = [nCH_in, nMic, [ksz_freq, ksz_time], [upsample_freq, upsample_time], [pad_freq, pad_time] ]

            for r in range(int(math.log2(self.ds_rate))-1):
                dconv = dcnn.ComplexConvWrapper(nn.ConvTranspose2d, *cfg_additional_layer, bias=False) # does not need bias due to BN
                self.decoders_additional.append(dconv)

                if(use_bn):
                    bn = dcnn.ComplexBatchNorm(nHidden)
                    self.decoders_additional.append(bn)

                act = dcnn.CLeakyReLU(0.01, inplace=True)
                self.decoders_additional.append(act)

            dconv = dcnn.ComplexConvWrapper(nn.ConvTranspose2d, *cfg_additional_last, bias=True)
            self.decoders_additional.append(dconv)

        self.ratio_mask_type = 'realimag'


    def forward(self, xr, xi, td=None):
        input_real, input_imag = xr, xi

        if(self.input_type == 'complex_ratio'):
            xr, xi = self.complex_ratio(xr, xi)
        elif(self.input_type == 'log_complex_ratio'):
            xr, xi = self.log_complex_ratio(xr, xi)
        elif(self.input_type == 'log_complex_ratio_cos'):
            xr, xi = self.log_complex_ratio(xr, xi, apply_cos=True)

        if(self.ds_rate > 1):
            xr, xi = self.downsample_freq(xr, xi)

        #pdb.set_trace()
        if(self.CW_freq > 0):
            xr = self.freq_splicing(xr, self.CW_freq)
            xi = self.freq_splicing(xi, self.CW_freq)

        xr = xr.transpose(1, 2) # NxMxFxT --> NxFxMxT
        xi = xi.transpose(1, 2)  # NxMxFxT --> NxFxMxT

        for layer in self.lcn: # complexdepthwiseConv-bn-leakyReLU
            xr, xi = layer(xr, xi)

        # transpose back
        xr = xr.transpose(1, 2) # NxFxMxT --> NxMxFxT
        xi = xi.transpose(1, 2) # NxFxMxT --> NxMxFxT

        if(self.ds_rate > 1): # complexConvtransposed-bn-leakyReLU
            for layer in self.decoders_additional:
                xr, xi = layer(xr, xi)

        #xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag) # no padding during forward

        if(self.reverb_frame > 1):
            # step 1) zero padding as much as reverb time
            input_real = Func.pad(input_real, (0, self.reverb_frame, 0, 0))  # (Ts,Te, Fs, Fe)
            input_imag = Func.pad(input_imag, (0, self.reverb_frame, 0, 0))  # (Ts,Te, Fs,Fe)

            # step 2) transpose
            input_real = input_real.transpose(0, 3).contiguous() # (NxMxFx(T+Lh)) --> ((T+Lh)xMxFxN)
            input_imag = input_imag.transpose(0, 3).contiguous() # (NxMxFx(T+Lh)) --> ((T+Lh)xMxFxN)

            # step 3) splicing
            input_real = self.frame_splicing(input_real)
            input_imag = self.frame_splicing(input_imag)

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

        ratio_mask_fn = self.get_ratio_mask(xr, xi)
        return ratio_mask_fn(input_real, input_imag)


    def get_ratio_mask(self, outr, outi):
        def inner_fn(r, i):
            if self.ratio_mask_type == 'realimag':
                return torch.sum(outr*r - outi*i, dim=1), torch.sum(outr*i + outr*i, dim=1), outr, outi

            elif self.ratio_mask_type == 'magphs':
                mag_mask = torch.sqrt(outr**2 + outi**2)
                phase_rotate = torch.atan2(outi, outr)

                mag = mag_mask * torch.sqrt(r**2 + i**2)
                phase = phase_rotate + torch.atan2(i, r)

                #return mag * torch.cos(phase), mag * torch.sin(phase) # single-mic
                return torch.sum(mag * torch.cos(phase), dim=1), torch.sum(mag * torch.sin(phase), dim=1), mag_mask, phase_rotate # multi-mic

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

    def frame_splicing(self, x):
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

    def freq_splicing(self, input, cw_side):
        N, H, F, T = input.size()
        cw_full = cw_side*2+1

        #output = torch.FloatTensor(N, H * cw_full, F, T).zero_()
        output = torch.FloatTensor(F, H * cw_full, N, T).zero_()

        #pdb.set_trace()

        input = Func.pad(input, (0, 0, cw_side, cw_side), mode='replicate')
        input = input.transpose(0, 2).contiguous() # NxHxFxT --> FxHxNxT

        fstart = 0
        fend = cw_full # +1 for python indexing
        for f in range(F):
            # ver 1: NxHxFxT
            #input_f = input[:, :, fstart:fend, :]
            #output[:, :, f, :] = input_f.view(N, -1, T)

            # ver 2: FxHxNxT
            input_f = input[fstart:fend, :, :, :]
            output[f, :, :, :] = input_f.view(-1, N, T)

            fstart += 1
            fend += 1

        #pdb.set_trace()
        output = output.transpose(0, 2) # FxHxNxT --> NxHxFxT

        '''
        for f in range(F):
            if (f >= cw_side and f <= F - 1 - cw_side):
                fstart = f - cw_side
                fend = f + cw_side
                input_f = input[:, fstart:fend + 1, :, :]
            elif (f < cw_side):
                # print('case 1')
                fstart = 0
                fend = f + cw_side
                nRepeat = cw_side - f
                # ver 1: torch.cat & torch.repeat
                #input_f = torch.cat((input[:, 0:1, :, :].repeat(1, nRepeat, 1, 1), input[:, fstart:fend + 1, :, :]), dim=1)

                # ver 2: F.pad
                input_f = Func.pad(input[:, fstart:fend + 1, :, :], (0, 0, nRepeat, 0, 0, 0, 0, 0), mode='replicate')

            elif (f > F - 1 - cw_side):
                fstart = f - cw_side
                fend = F - 1
                nRepeat = f - F + cw_side + 1
                # ver 1: torch.cat & torch.repeat
                #input_f = torch.cat((input[:, fstart:fend + 1, :, :], input[:, F - 1:F, :, :].repeat(1, nRepeat, 1, 1)), dim=1)

                # ver 2: F.pad
                input_f = Func.pad(input[:, fstart:fend + 1, :, :], (0, 0, 0, nRepeat, 0, 0, 0, 0), mode = 'replicate')

            output[:, f, :, :] = input_f.view(output.size(0), -1, output.size(3))
            '''

        return output.cuda()

    def complex_ratio(self, xr, xi):
        # xr, xr: NxMxFxT
        eps = 1e-10
        N, M, F, T = xr.size()
        yr, yi = torch.FloatTensor(N, M-1, F, T).cuda(), torch.FloatTensor(N, M-1, F, T).cuda()
        ref_pow = (xr[:, 0]*xr[:, 0] + xi[:, 0]*xi[:, 0] + eps)
        for m in range(M-1):
            yr[:, m] = (xr[:, m+1]*xr[:, 0] + xi[:, m+1]*xi[:, 0])/ref_pow
            yi[:, m] = (xi[:, m+1]*xr[:, 0] - xr[:, m+1]*xi[:, 0])/ref_pow

        return yr, yi

    def log_complex_ratio(self, xr, xi, apply_cos=False):
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

        if(apply_cos):
            yi = torch.cos(yi)

        return yr, yi
