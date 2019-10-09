import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np
import scipy.io as sio
import pdb

def sInvSDR_time(clean, clean_est, eps=1e-12): # scale invariant SDR loss function
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
    correlation = bsum(clean * clean_est)
    energies = torch.norm(clean, p=2, dim=1) * torch.norm(clean_est, p=2, dim=1)

    numerator = correlation*correlation
    denominator = energies*energies - correlation*correlation

    sInvSDR = 10*torch.log10(numerator/(denominator + eps))

    return sInvSDR # #minibatchx1

def SD_SDR_complex_ipd(clean_real, clean_imag, out_real, out_imag, Tlist, eps=1e-12): # scale invariant SDR loss function
    # Tlist as dummy variable

    inner_product = torch.sum(torch.sum(clean_real*out_real + clean_imag*out_imag, dim=2), dim=1) # Re(x^Hy)
    power_clean = torch.sum(torch.sum(clean_real*clean_real + clean_imag*clean_imag, dim=2), dim=1)
    power_output = torch.sum(torch.sum(out_real*out_real + out_imag*out_imag, dim=2), dim=1)

    inner_product_sq = inner_product*inner_product
    power_clean_output = power_clean*power_output

    numerator = inner_product_sq
    denominator = power_clean_output - inner_product_sq

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #return torch.mean(wSDR) # scalar
    return sInvSDR # #

def SD_SDR_spec_RIconcat(clean_real, clean_imag, out_real, out_imag, Tlist, eps=1e-12): # scale invariant SDR loss function
    # Tlist as dummy variable

    # concat real & imag: {(NxFxT), (NxFxT)} --> (Nx2FxT)

    clean = torch.cat((clean_real, clean_imag), dim=1)
    out = torch.cat((out_real, out_imag), dim=1)

    inner_product = torch.sum(torch.sum(clean*out, dim=2), dim=1) # Re(x^Hy)
    power_clean = torch.sum(torch.sum(clean*clean, dim=2), dim=1)
    power_output = torch.sum(torch.sum(out*out, dim=2), dim=1)

    inner_product_sq = inner_product*inner_product
    power_clean_output = power_clean*power_output

    numerator = inner_product_sq
    denominator = power_clean_output - inner_product_sq

    SDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #return torch.mean(wSDR) # scalar
    return SDR # #minibatchx1

def var_time(W):
    # time dimension in W: 3
    var = torch.var(W, dim=3).mean()

    return var

def refIR_demix_positive(X_real, X_imag, W_real, W_imag, S_real, S_imag, Tlist, eps=1e-20):
    F = X_real.size(2)

    S_mag = torch.sqrt(S_real*S_real + S_imag*S_imag+eps)

    # size: NxMxFxT
    O_real = torch.sum(W_real*X_real-W_imag*X_imag, dim=1) # NxFxT
    O_imag = torch.sum(W_real*X_imag+W_imag*X_real, dim=1) # NxFxT

    O_mag = torch.sqrt(O_real*O_real + O_imag*O_imag + eps) # NxFxT

    distortion_mag =  O_mag-S_mag
    distortion_power = torch.sum(torch.sum(distortion_mag*distortion_mag, dim=2), dim=1)

    Tlist_float = Tlist.float().cuda()

    distortion_power_frame_normalized_negative = -distortion_power/(Tlist_float*F) # x(-1) for -loss convention

    return distortion_power_frame_normalized_negative

def reference_position_demixing(X_real, X_imag, W_real, W_imag, Tlist):
    F = X_real.size(2)

    # size: NxMxFxT
    XW_real = torch.sum(W_real*X_real-W_imag*X_imag, dim=1) # NxFxT
    XW_imag = torch.sum(W_real*X_imag+W_imag*X_real, dim=1) # NxFxT

    XW_power = torch.sum(torch.sum(XW_real*XW_real + XW_imag*XW_imag, dim=2), dim=1) # Nx1

    Tlist_float = Tlist.float().cuda()
    XW_power_frame_normalized_negative = -XW_power/(Tlist_float*F) # x(-1) for -loss convention

    return XW_power_frame_normalized_negative

def reference_position_demixing_pow(X_real, X_imag, W_real, W_imag, Tlist):
    F = X_real.size(2)

    # size: NxMxFxT
    XW_real = torch.sum(W_real*X_real-W_imag*X_imag, dim=1) # NxFxT
    XW_imag = torch.sum(W_real*X_imag+W_imag*X_real, dim=1) # NxFxT

    XW_power = XW_real*XW_real + XW_imag*XW_imag # NxFxT
    XW_power_pow4 = torch.sum(torch.sum(XW_power*XW_power, dim=2), dim=1) # Nx1

    Tlist_float = Tlist.float().cuda()
    XW_power_frame_normalized_negative = -XW_power_pow4/(Tlist_float*F) # x(-1) for -loss convention

    return XW_power_frame_normalized_negative

def distortion_em_mag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-12):
    F = clean_real.size(1)

    clean_mag = torch.sqrt(clean_real*clean_real + clean_imag*clean_imag+eps)
    output_mag = torch.sqrt(output_real*output_real + output_imag*output_imag+eps)

    distortion_mag =  output_mag-clean_mag
    distortion_power = torch.sum(torch.sum(distortion_mag*distortion_mag, dim=2), dim=1)

    Tlist_float = Tlist.float().cuda()

    distortion_power_frame_normalized_negative = -distortion_power/(Tlist_float*F) # x(-1) for -loss convention

    return distortion_power_frame_normalized_negative

def distortion_em_pow(clean_real, clean_imag, output_real, output_imag, Tlist):
    F = clean_real.size(1)

    clean_pow = clean_real*clean_real + clean_imag*clean_imag # use pow instead of mag to avoid backprop through sqrt()
    output_pow = output_real*output_real + output_imag*output_imag # use pow instead of mag to avoid backprop through sqrt()

    distortion_pow = output_pow-clean_pow
    err = torch.sum(torch.sum(distortion_pow*distortion_pow, dim=2), dim=1)

    Tlist_float = Tlist.float().cuda()

    distortion_power_frame_normalized_negative = -err/(Tlist_float*F) # x(-1) for -loss convention

    return distortion_power_frame_normalized_negative

def distortion_em_L1(clean_real, clean_imag, output_real, output_imag, Tlist):
    F = clean_real.size(1)

    clean_pow = clean_real*clean_real + clean_imag*clean_imag # use pow instead of mag to avoid backprop through sqrt()
    output_pow = output_real*output_real + output_imag*output_imag # use pow instead of mag to avoid backprop through sqrt()

    distortion_pow = output_pow-clean_pow
    err = torch.sum(torch.sum(abs(distortion_pow), dim=2), dim=1)
    #err = torch.sum(torch.sum(distortion_pow * distortion_pow, dim=2), dim=1)

    Tlist_float = Tlist.float().cuda()

    distortion_power_frame_normalized_negative = -err/(Tlist_float*F) # x(-1) for -loss convention

    return distortion_power_frame_normalized_negative

def Wdiff_mag(Wgt_real, Wgt_imag, West_real, West_imag, Tlist, eps=1e-20):
    F = Wgt_real.size(2)

    Wgt_mag = torch.sqrt(Wgt_real*Wgt_real + Wgt_imag*Wgt_imag+eps)
    West_mag = torch.sqrt(West_real*West_real + West_imag*West_imag+eps)

    err = Wgt_mag - West_mag # NxMxFxT

    Tlist_float = Tlist.float().cuda()

    MSE = torch.sum(torch.sum(torch.sum(err*err, dim=3), dim=2), dim=1)/(Tlist_float*F)

    return -MSE # x(-1) for -loss convention

def Wdiff_mag_gtnormalized(Wgt_real, Wgt_imag, West_real, West_imag, Tlist, eps=1e-20):
    Wgt_mag = torch.sqrt(Wgt_real*Wgt_real + Wgt_imag*Wgt_imag+eps)
    West_mag = torch.sqrt(West_real*West_real + West_imag*West_imag+eps)

    err = Wgt_mag - West_mag # NxMxFxT

    #Tlist_float = Tlist.float().cuda()

    Wgt_sq_sum = torch.sum(torch.sum(torch.sum(Wgt_mag*Wgt_mag, dim=3), dim=2), dim=1)

    distortion_to_signal_power = torch.sum(torch.sum(torch.sum(err*err, dim=3), dim=2), dim=1)/Wgt_sq_sum
    # note that distortion_to_signal_power is already normalized by #frame

    return -distortion_to_signal_power# x(-1) for -loss convention


def Wdiff_realimag_gtnormalized(Wgt_real, Wgt_imag, West_real, West_imag, Tlist, eps=1e-20):
    Wgt = torch.cat((Wgt_real, Wgt_imag), dim=1) # concat along mic dimension
    West = torch.cat((West_real, West_imag), dim=1) # concat along mic dimension

    err = Wgt - West # NxMxFxT

    Wgt_sq_sum = torch.sum(torch.sum(torch.sum(Wgt*Wgt, dim=3), dim=2), dim=1)
    distortion_to_signal_power = torch.sum(torch.sum(torch.sum(err*err, dim=3), dim=2), dim=1)/Wgt_sq_sum
    # note that distortion_to_signal_power is already normalized by #frame

    return -distortion_to_signal_power# x(-1) for -loss convention

def Wdiff_realimag(Wgt_real, Wgt_imag, West_real, West_imag, Tlist):
    F = Wgt_real.size(2)

    Wgt = torch.cat((Wgt_real, Wgt_imag), dim=1) # concat along mic dimension
    West = torch.cat((West_real, West_imag), dim=1) # concat along mic dimension

    err = Wgt - West # NxMxFxT
    #print('||Wgt|| = ' + str(Wgt.norm().item()))

    Tlist_float = Tlist.float().cuda()

    MSE = torch.sum(torch.sum(torch.sum(err*err, dim=3), dim=2), dim=1)/(Tlist_float*F)

    return -MSE # x(-1) for -loss convention

def SDR_Wdiff_realimag(Wgt_real, Wgt_imag, West_real, West_imag, Tlist, eps=1e-20):
    # Tlist is for dummy

    Wgt = torch.cat((Wgt_real, Wgt_imag), dim=1) # concat along mic dimension
    West = torch.cat((West_real, West_imag), dim=1) # concat along mic dimension

    distortion = Wgt - West # NxMxFxT

    signal_power = torch.sum(torch.sum(torch.sum(Wgt*Wgt, dim=3), dim=2), dim=1)
    distortion_power = torch.sum(torch.sum(torch.sum(distortion*distortion, dim=3), dim=2), dim=1)

    SDR = 10*(torch.log10(signal_power+eps)-torch.log10(distortion_power+eps))
    # note that SDR is already normalized by #frame

    return SDR

def SDR_Wdiff_mag(Wgt_real, Wgt_imag, West_real, West_imag, Tlist, eps=1e-20):
    # Tlist is for dummy

    Wgt_mag = torch.sqrt(Wgt_real*Wgt_real + Wgt_imag*Wgt_imag + eps)
    West_mag = torch.sqrt(West_real*West_real + West_imag*West_imag + eps)

    distortion = Wgt_mag - West_mag # NxMxFxT

    signal_power = torch.sum(torch.sum(torch.sum(Wgt_mag*Wgt_mag, dim=3), dim=2), dim=1)
    distortion_power = torch.sum(torch.sum(torch.sum(distortion*distortion, dim=3), dim=2), dim=1)

    SDR = 10*(torch.log10(signal_power+eps)-torch.log10(distortion_power+eps))
    # note that SDR is already normalized by #frame

    return SDR

def Wdiff_pow(Wgt_real, Wgt_imag, West_real, West_imag, Tlist):
    F = Wgt_real.size(2)

    Wgt_pow = Wgt_real*Wgt_real + Wgt_imag*Wgt_imag # use pow instead of mag to avoid backprop through sqrt()
    West_pow = West_real*West_real + West_imag*West_imag  # use pow instead of mag to avoid backprop through sqrt()

    err = Wgt_pow - West_pow # NxMxFxT

    Tlist_float = Tlist.float().cuda()

    MSE = torch.sum(torch.sum(torch.sum(err*err, dim=3), dim=2), dim=1)/(Tlist_float*F)

    return -MSE # x(-1) for -loss convention

def Wdiff_L1(Wgt_real, Wgt_imag, West_real, West_imag, Tlist):
    F = Wgt_real.size(2)

    Wgt_pow = Wgt_real*Wgt_real + Wgt_imag*Wgt_imag # use pow instead of mag to avoid backprop through sqrt()
    West_pow = West_real*West_real + West_imag*West_imag  # use pow instead of mag to avoid backprop through sqrt()

    err = Wgt_pow - West_pow # NxMxFxT

    Tlist_float = Tlist.float().cuda()

    MSE = torch.sum(torch.sum(torch.sum(abs(err), dim=3), dim=2), dim=1)/(Tlist_float*F)

    return -MSE # x(-1) for -loss convention

def distortion_em_realimag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-10):
    distortion_real = output_real-clean_real
    distortion_imag = output_imag-clean_imag
    distortion_power = torch.sum(torch.sum(distortion_real*distortion_real + distortion_imag*distortion_imag, dim=2), dim=1)

    Tlist_float = Tlist.float().cuda()

    distortion_power_frame_normalized_negative = -distortion_power/Tlist_float # x(-1) for -loss convention

    return distortion_power_frame_normalized_negative

def SDR_em_realimag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-10):  # scale invariant SDR loss function
    # Tlist as dummy variable

    signal_power = torch.sum(torch.sum(clean_real*clean_real + clean_imag*clean_imag, dim=2), dim=1)

    distortion_real = output_real-clean_real
    distortion_imag = output_imag-clean_imag
    distortion_power = torch.sum(torch.sum(distortion_real*distortion_real + distortion_imag*distortion_imag, dim=2), dim=1)

    SDR = 10*(torch.log10(signal_power+eps) - torch.log10(distortion_power+eps))

    return SDR # #minibatchx1


def SDR_em_mag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-10):  # scale invariant SDR loss function
    # Tlist as dummy variable

    clean_mag = torch.sqrt(clean_real*clean_real + clean_imag*clean_imag+eps)
    output_mag = torch.sqrt(output_real*output_real + output_imag*output_imag+eps)
    signal_power = torch.sum(torch.sum(clean_mag*clean_mag, dim=2), dim=1)

    distortion_mag = output_mag-clean_mag
    distortion_power = torch.sum(torch.sum(distortion_mag*distortion_mag, dim=2), dim=1)

    SDR = 10*(torch.log10(signal_power+eps) - torch.log10(distortion_power+eps))

    return SDR # #minibatchx1

def sInvSDR_mag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-10):  # scale invariant SDR loss function
    # Tlist as dummy variable

    # add eps for gradient explosion at sqrt(x)=0
    clean_pow = clean_real*clean_real + clean_imag*clean_imag + eps # NxFxT
    '''
    if(len(output_list) == 2):
        output_real = output_list[0]
        output_imag = output_list[1]
        output_pow = output_real*output_real + output_imag*output_imag + eps # NxFxT
    elif(len(output_list) == 1):
        output_pow = output_list[0] + eps # NxFxT
    '''
    output_pow = output_real*output_real + output_imag*output_imag + eps # NxFxT

    cdim = clean_pow.dim()
    if(cdim == 3):
        clean_pow_L2 = torch.sum(torch.sum(clean_pow, dim=2), dim=1)
        output_pow_L2 = torch.sum(torch.sum(output_pow, dim=2), dim=1)
        inner_product = torch.sum(torch.sum(torch.sqrt(clean_pow)*torch.sqrt(output_pow), dim=2), dim=1) # Re(x^Hy)
    elif(cdim == 2):
        # minibatch size = 1
        clean_pow_L2 = torch.sum(torch.sum(clean_pow, dim=1), dim=0)
        output_pow_L2 = torch.sum(torch.sum(output_pow, dim=1), dim=0)
        inner_product = torch.sum(torch.sum(torch.sqrt(clean_pow)*torch.sqrt(output_pow), dim=1), dim=0) # Re(x^Hy)

    power = clean_pow_L2*output_pow_L2
    inner_prod_sq = inner_product*inner_product

    numerator = inner_prod_sq
    denominator = power - inner_prod_sq

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))


    # sio.savemat('SD-SDR.mat', {'clean_pow_L2': clean_pow_L2.data.cpu().numpy(),
    #                            'output_pow_L2': output_pow_L2.data.cpu().numpy(),
    #                            'power': power.data.cpu().numpy(),
    #                            'inner_product': inner_product.data.cpu().numpy(),
    #                            'sInvSDR': sInvSDR.data.cpu().numpy()})

    # sio.savemat('SD-SDR.mat', {'sInvSDR': sInvSDR.data.cpu().numpy()})

    #return torch.mean(wSDR) # scalar
    return sInvSDR # #minibatchx1

def srcIndepSDR_Cproj_by_WH(Wreal, Wimag, Hreal, Himag, Tlist, eps=1e-6):  # scale invariant SDR loss function
    # H, W: NxMxFxT
    # Tlist: Nx1
    N, M, F, Tmax = Wreal.size()

    Cr = torch.sum(Wreal*Hreal-Wimag*Himag, dim=1) # NxFxT
    Ci = torch.sum(Wreal*Himag+Wimag*Hreal, dim=1) # NxFxT

    # sio.savemat('SI-WH.mat', {'Wreal':Wreal.data.cpu().numpy(),
    #                           'Hreal':Hreal.data.cpu().numpy(),
    #                              'Wimag':Wimag.data.cpu().numpy(),
    #                            'Himag':Himag.data.cpu().numpy(),
    #                              'Cr':Cr.data.cpu().numpy(),
    #                              'Ci':Ci.data.cpu().numpy()})

    # Since Wreal, Wimag has zero value at garbage frame, Cr, Ci also has zero value at garbage frame

    Cmag = torch.sqrt(Cr*Cr + Ci*Ci + eps)

    #Cref = torch.sqrt(1/(Tlist*F)).unsqueeze(1).unsqueeze(2).expand_as(Cmag) # Nx1 --> NxFxT (don't need expand_as)
    Cref = torch.sqrt(1 / (Tlist.float().cuda() * F)).unsqueeze(1).unsqueeze(2) # Nx1 --> Nx1x1

    # make garbage frame zero
    for n in range(N):
        t = Tlist[n]
        Cref[:, :, t:] = 0

    # project to Cmag to Cref (L2-norm=1)
    inner_prod = torch.sum(torch.sum(Cmag*Cref, dim=2), dim=1) # (NxFxT)*(Nx1x1)
    Ctarget = inner_prod.unsqueeze(1).unsqueeze(2)*Cref  # (Nx1x1)*(Nx1x1) = (Nx1x1)
    Cdistortion = Cmag-Ctarget # (NxFxT)-(Nx1x1) = (NxFxT)

    Ctarget_pow = inner_prod*inner_prod # Nx1
    Cdistortion_pow = torch.sum(torch.sum(Cdistortion*Cdistortion, dim=2), dim=1) # Nx1

    sInvSDR = 10*(torch.log10(Ctarget_pow + eps) - torch.log10(Cdistortion_pow + eps))

    return sInvSDR  #minibatchx1

def srcIndepSDR_Cproj_by_SShat(clean_real, clean_imag, out_real, out_imag, Tlist, eps=1e-20):  # scale invariant SDR loss function
    # H, W: NxMxFxT
    # Tlist: Nx1
    N, F, Tmax = clean_real.size()

    cleanSTFT_pow = clean_real * clean_real + clean_imag * clean_imag + eps
    Cr= (out_real* clean_real + out_imag * clean_imag) / cleanSTFT_pow
    Ci = (-out_real * clean_imag + out_imag * clean_real) / cleanSTFT_pow

    # Since Wreal, Wimag has zero value at garbage frame, Cr, Ci also has zero value at garbage frame
    Cmag = torch.sqrt(Cr*Cr + Ci*Ci + eps)

    #pdb.set_trace()

    # Ver 1. SDR value different from matlab & online version
    '''
    Cref = torch.sqrt(1 / (Tlist.float().cuda() * F)).unsqueeze(1).unsqueeze(2) # Nx1 --> Nx1x1
    for n in range(N):
        t = Tlist[n]
        Cref[:, :, t:].fill_(0)
    '''

    # Ver 2. correct & memory efficient
    # 2-1. cannot be backpropable (inplace operation)
    '''
    Ctarget_scale = torch.FloatTensor(N).cuda() #--> cannot be backpropable (inplace operation)
    for n in range(N):
        t = Tlist[n]
        Ctarget_scale[n] = torch.sum(torch.sum(Cmag[n, :, :t], dim=1), dim=0)/(t*F)
        #Ctarget_pow[n] = Ctarget_scale[n]*Ctarget_scale[n]*t*F
    '''
    # 2-2.
    Tlist_float = Tlist.float().cuda()

    # use of below operation & sqrt makes Cmag not back-propable (in-place operation)
    '''
    for n in range(N):
        t = Tlist[n]
        #Cmag[n, :, t:].fill_(0)
        Cmag[n, :, t:] = 0
    '''
    Ctarget_scale = torch.sum(torch.sum(Cmag, dim=2), dim=1)/(Tlist_float*F)

    Cdistortion = Cmag-Ctarget_scale.unsqueeze(1).unsqueeze(2)
    Ctarget_pow = Ctarget_scale*Ctarget_scale*Tlist_float*F

    # make garbage frame zero
    for n in range(N):
        t = Tlist[n]
        #Cdistortion[n, :, t:].fill_(0)
        Cdistortion[n, :, t:] = 0

    # project to Cmag to Cref (L2-norm=1)
    #inner_prod = torch.sum(torch.sum(Cmag*Cref, dim=2), dim=1) # sum((NxFxT)*(Nx1x1)) = Nx1
    #Ctarget = inner_prod.unsqueeze(1).unsqueeze(2)*Cref  # (Nx1x1)*(Nx1x1) = (Nx1x1)
    #Cdistortion = Cmag-Ctarget # (NxFxT)-(Nx1x1) = (NxFxT)

    #Ctarget_pow = inner_prod*inner_prod # Nx1
    Cdistortion_pow = torch.sum(torch.sum(Cdistortion*Cdistortion, dim=2), dim=1) # Nx1

    sInvSDR = 10*(torch.log10(Ctarget_pow + eps) - torch.log10(Cdistortion_pow + eps))

    #sio.savemat('SI-SShat.mat', {'cleanSTFT_pow':cleanSTFT_pow.data.cpu().numpy(),
#                                   'out_real':out_real.data.cpu().numpy(),
#                                   'out_imag':out_imag.data.cpu().numpy(),
#                                  'Cr':Cr.data.cpu().numpy(),
#                                  'Ci':Ci.data.cpu().numpy(),
#                                  'sInvSDR':sInvSDR.data.cpu().numpy()})

    return sInvSDR  #minibatchx1


def SI_SDR_spec_RIconcat(clean_real, clean_imag, out_real, out_imag, Tlist, eps=1e-12):  # scale invariant SDR loss function
    # H, W: NxMxFxT
    # Tlist: Nx1
    N, F, Tmax = clean_real.size()

    cleanSTFT_pow = clean_real * clean_real + clean_imag * clean_imag + eps
    Cr = (out_real * clean_real + out_imag * clean_imag) / cleanSTFT_pow
    Ci = (-out_real * clean_imag + out_imag * clean_real) / cleanSTFT_pow

    # concat real & imag
    C = torch.cat((Cr, Ci), dim=1)

    #pdb.set_trace()

    #Cref = torch.sqrt(1/(Tlist*F)).unsqueeze(1).unsqueeze(2).expand_as(Cmag) # Nx1 --> NxFxT (don't need expand_as)
    Cref = torch.sqrt(1 / (Tlist.float().cuda() * F*2)).unsqueeze(1).unsqueeze(2) # Nx1 --> Nx1x1

    # make garbage frame zero
    for n in range(N):
        t = Tlist[n]
        Cref[:, :, t:] = 0

    # project to Cmag to Cref (L2-norm=1)
    inner_prod = torch.sum(torch.sum(C*Cref, dim=2), dim=1) # sum((Nx2FxT)*(Nx1x1)) = Nx1
    Ctarget = inner_prod.unsqueeze(1).unsqueeze(2)*Cref  # (Nx1x1)*(Nx1x1) = (Nx1x1)
    Cdistortion = C-Ctarget # (Nx2FxT)-(Nx1x1) = (Nx2FxT)

    Ctarget_pow = inner_prod*inner_prod # Nx1
    Cdistortion_pow = torch.sum(torch.sum(Cdistortion*Cdistortion, dim=2), dim=1) # Nx1

    SDR = 10*(torch.log10(Ctarget_pow + eps) - torch.log10(Cdistortion_pow + eps))

    # sio.savemat('SI-SShat.mat', {'cleanSTFT_pow':cleanSTFT_pow.data.cpu().numpy(),
    #                               'out_real':out_real.data.cpu().numpy(),
    #                               'out_imag':out_imag.data.cpu().numpy(),
    #                              'Cr':Cr.data.cpu().numpy(),
    #                              'Ci':Ci.data.cpu().numpy(),
    #                              'sInvSDR':sInvSDR.data.cpu().numpy()})

    return SDR  #minibatchx1


def SI_SDR_complex_ipd(clean_real, clean_imag, out_real, out_imag, Tlist, eps=1e-12):  # scale invariant SDR loss function
    # H, W: NxMxFxT
    # Tlist: Nx1
    N, F, Tmax = clean_real.size()

    cleanSTFT_pow = clean_real * clean_real + clean_imag * clean_imag + eps
    Chat_r = (out_real * clean_real + out_imag * clean_imag) / cleanSTFT_pow
    Chat_i = (-out_real * clean_imag + out_imag * clean_real) / cleanSTFT_pow

    # concat real & imag
    #Chat = torch.cat((Chat_r, Chat_i), dim=1)

    C = torch.sqrt(1 / (Tlist.float().cuda() * F)).unsqueeze(1).unsqueeze(2) # Nx1 --> Nx1x1

    # make garbage frame zero
    for n in range(N):
        t = Tlist[n]
        C[:, :, t:] = 0

    # project to Cmag to Cref (L2-norm=1)
    inner_prod = torch.sum(torch.sum(Chat_r*C, dim=2), dim=1) # (Chat_r*C_r+Chat_i*C_i), C_i=0, sum((NxFxT)*(Nx1x1)) = Nx1
    Ctarget = inner_prod.unsqueeze(1).unsqueeze(2)*C  # (Nx1x1)*(Nx1x1) = (Nx1x1)
    Cdistortion = C-Ctarget # (Nx2FxT)-(Nx1x1) = (Nx2FxT)

    Ctarget_pow = inner_prod*inner_prod # Nx1
    Cdistortion_pow = torch.sum(torch.sum(Cdistortion*Cdistortion, dim=2), dim=1) # Nx1

    SDR = 10*(torch.log10(Ctarget_pow + eps) - torch.log10(Cdistortion_pow + eps))

    # sio.savemat('SI-SShat.mat', {'cleanSTFT_pow':cleanSTFT_pow.data.cpu().numpy(),
    #                               'out_real':out_real.data.cpu().numpy(),
    #                               'out_imag':out_imag.data.cpu().numpy(),
    #                              'Cr':Cr.data.cpu().numpy(),
    #                              'Ci':Ci.data.cpu().numpy(),
    #                              'sInvSDR':sInvSDR.data.cpu().numpy()})

    return SDR  #minibatchx1


def cossim_time(clean, clean_est, eps=1e-10): # scale invariant SDR loss function
    bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
    def mSDRLoss(orig, est):
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return correlation / (energies + eps)

    cossim = mSDRLoss(clean, clean_est)
    #return torch.mean(wSDR) # scalar
    return cossim # #minibatchx1
