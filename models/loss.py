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

def reference_position_demixing(refmic_real, refmic_imag, mask_real, mask_imag, Tlist):
    # size: NxMxFxT
    weighted_sum_real = torch.sum(mask_real*refmic_real-mask_imag*refmic_imag, dim=1) # NxFxT
    weighted_sum_imag = torch.sum(mask_real*refmic_imag+mask_imag*refmic_real, dim=1) # NxFxT

    weighted_sum_power = torch.sum(torch.sum(weighted_sum_real*weighted_sum_real + weighted_sum_imag*weighted_sum_imag, dim=2), dim=1) # Nx1

    Tlist_float = Tlist.float().cuda()
    weighted_sum_power_frame_normalized_negative = -weighted_sum_power/Tlist_float # x(-1) for -loss convention

    return weighted_sum_power_frame_normalized_negative

def distortion_em_mag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-10):
    clean_mag = torch.sqrt(clean_real*clean_real + clean_imag*clean_imag+eps)
    output_mag = torch.sqrt(output_real*output_real + output_imag*output_imag+eps)

    distortion_mag =  output_mag-clean_mag
    distortion_power = torch.sum(torch.sum(distortion_mag*distortion_mag, dim=2), dim=1)

    Tlist_float = Tlist.float().cuda()

    distortion_power_frame_normalized_negative = -distortion_power/Tlist_float # x(-1) for -loss convention

    return distortion_power_frame_normalized_negative

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

def srcIndepSDR_mag(Wreal, Wimag, Hreal, Himag, Tlist, eps=1e-8):  # scale invariant SDR loss function

    # W: NxMxFxT
    # H: NxMxFxT
    # Tlist: Nx1

    # Convert tensors to time-independent by averaging(NxMxF)
    F = Wreal.size(2)
    Tlist = Tlist.unsqueeze(1).unsqueeze(2).float().cuda()

    Wreal = Wreal.sum(3)/Tlist
    Wimag = Wimag.sum(3)/Tlist
    Hreal = Hreal.sum(3)/Tlist
    Himag = Himag.sum(3)/Tlist

    Hreal.requires_grad = False
    Himag.requires_grad = False

    # Get weighted sum 'across mic'
    # no mic dimension summation
    Cr = Wreal*Hreal-Wimag*Himag
    Ci = Wreal*Himag+Wimag*Hreal

    # mic dimension summation
    Cr = Cr.sum(1) # NxMxF --> NxF
    Ci = Ci.sum(1)  # NxMxF --> NxF

    # Get magnitude
    Cmag = torch.sqrt(Cr*Cr + Ci*Ci) # NxF

    # Get sum & sqsum over frequency
    Cmag_sum = Cmag.sum(1) # NxF --> Nx1
    Cmag_sqsum = Cmag.pow(2).sum(1) # NxF --> Nx1

    # Get SDR per samples
    inner_prod_sq = Cmag_sum*Cmag_sum
    numerator = inner_prod_sq
    denominator = F*Cmag_sqsum - inner_prod_sq

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #sInvSDR_mean = torch.mean(sInvSDR).item()
    #print(sInvSDR_mean)

    #return sInvSDR, Cmag  #minibatchx1
    return sInvSDR  # minibatchx1


def srcIndepSDR_mag_diffperT(Wreal, Wimag, Hreal, Himag, Tlist, eps=1e-10):  # scale invariant SDR loss function

    # W: NxMxFxT
    # H: NxMxFxT
    # Tlist: Nx1

    Hreal.requires_grad = False
    Himag.requires_grad = False

    # Convert tensors to time-independent by averaging(NxMxF)
    F = Wreal.size(2)
    Tlist = Tlist.float().cuda()
    '''
    Wreal = Wreal.sum(3)/Tlist
    Wimag = Wimag.sum(3)/Tlist
    Hreal = Hreal.sum(3)/Tlist
    Himag = Himag.sum(3)/Tlist
    '''
    # Get weighted sum 'across mic'
    # no mic dimension summation
    Cr = Wreal*Hreal-Wimag*Himag # NxMxFxT
    Ci = Wreal*Himag+Wimag*Hreal # NxMxFxT

    # mic dimension summation
    Cr = Cr.sum(1).squeeze(1) # NxMxFxT --> NxFxT
    Ci = Ci.sum(1).squeeze(1)  # NxMxFxT --> NxFxT

    # Get magnitude
    Cmag = torch.sqrt(Cr*Cr + Ci*Ci + eps) # NxFxT

    # Get sum & sqsum over frequency
    Cmag_sum = Cmag.sum(1).squeeze(1) # NxFxT --> NxT
    Cmag_sqsum = Cmag.pow(2).sum(1).squeeze(1) # NxFxT --> NxT

    # Get SDR per samples
    inner_prod_sq = Cmag_sum*Cmag_sum # NxT
    numerator = inner_prod_sq  # NxT
    denominator = F*Cmag_sqsum - inner_prod_sq # NxT

    # Mean across time
    numerator = numerator.sum(1)/Tlist
    denominator = denominator.sum(1)/Tlist

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #sInvSDR_mean = torch.mean(sInvSDR).item()
    #print(sInvSDR_mean)

    #return sInvSDR, Cmag  #minibatchx1
    return sInvSDR  # minibatchx1


def srcIndepSDR_freqpower(Wreal, Wimag, Hreal, Himag, Tlist, Pf, eps=1e-10):  # scale invariant SDR loss function


    # W: NxMxFxT
    # H: NxMxFxT
    # Pf: Fx1
    # Tlist: Nx1

    # power per freq
    #Pf = weights_per_freq*weights_per_freq # F
    #pdb.set_trace()
    #Pf_sum = Pf.sum() # = 1
    Pf_unsqueeze = Pf.unsqueeze(0) # 1xF

    # Convert tensors to time-independent by averaging(NxMxF)
    Tlist = Tlist.unsqueeze(1).unsqueeze(2).float().cuda()

    Wreal = Wreal.sum(3)/Tlist
    Wimag = Wimag.sum(3)/Tlist
    Hreal = Hreal.sum(3)/Tlist
    Himag = Himag.sum(3)/Tlist

    Hreal.requires_grad = False
    Himag.requires_grad = False


    # Get weighted sum 'across mic'
    # no mic dimension summation
    Cr = Wreal*Hreal-Wimag*Himag
    Ci = Wreal*Himag+Wimag*Hreal

    # mic dimension summation
    Cr = Cr.sum(1) # NxMxF --> NxF
    Ci = Ci.sum(1)  # NxMxF --> NxF

    # Get magnitude
    Cmag = torch.sqrt(Cr*Cr + Ci*Ci) # NxF

    # Get (sum & sqsum) * Pf
    Cmag_Pf = Cmag*Pf_unsqueeze # (NxF)*(1xF) = (NxF)
    Cmagsq_Pf = Cmag.pow(2)*Pf_unsqueeze # (NxF)*(1xF) --> (NxF)

    # Sum over frequency
    Cmag_Pf = Cmag_Pf.sum(1)
    Cmagsq_Pf = Cmagsq_Pf.sum(1)

    # Get SDR per samples
    inner_prod_sq = Cmag_Pf*Cmag_Pf
    numerator = inner_prod_sq
    #denominator = Pf_sum*Cmagsq_Pf - inner_prod_sq
    denominator = Cmagsq_Pf - inner_prod_sq  #Pf_sum = 1

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #sInvSDR_mean = torch.mean(sInvSDR).item()
    #print(sInvSDR_mean)

    #return sInvSDR, torch.zeros(1)  #minibatchx1, dummy tensor as a return
    return sInvSDR

def srcIndepSDR_freqpower_diffperT(Wreal, Wimag, Hreal, Himag, Tlist, Pf, eps=1e-10):  # scale invariant SDR loss function
    # W: NxMxFxT
    # H: NxMxFxT
    # Pf: Fx1
    # Tlist: Nx1

    Hreal.requires_grad = False
    Himag.requires_grad = False

    # power per freq
    #Pf = weights_per_freq*weights_per_freq # F
    #pdb.set_trace()
    #Pf_sum = Pf.sum() # = 1
    Pf_unsqueeze = Pf.unsqueeze(0).unsqueeze(2) # 1xFx1

    # Convert tensors to time-independent by averaging(NxMxF)
    Tlist = Tlist.float().cuda()

    '''
    Wreal = Wreal.sum(3)/Tlist
    Wimag = Wimag.sum(3)/Tlist
    Hreal = Hreal.sum(3)/Tlist
    Himag = Himag.sum(3)/Tlist
    '''

    # Get weighted sum 'across mic'
    # no mic dimension summation
    #pdb.set_trace()
    Cr = Wreal*Hreal-Wimag*Himag # NxMxFxT
    Ci = Wreal*Himag+Wimag*Hreal # NxMxFxT

    # mic dimension summation
    Cr = Cr.sum(1).squeeze(1) # NxMxFxT --> NxFxT
    Ci = Ci.sum(1).squeeze(1) # NxMxFxT --> NxFxT

    # Get magnitude
    Cmag = torch.sqrt(Cr*Cr + Ci*Ci + eps) # NxFxT

    # Get (sum & sqsum) * Pf
    Cmag_Pf = Cmag*Pf_unsqueeze # (NxFxT)*(1xFx1) = (NxFxT)
    Cmagsq_Pf = Cmag.pow(2)*Pf_unsqueeze # (NxFxT)*(1xFx1) --> (NxFxT)

    # Sum over frequency
    Cmag_Pf = Cmag_Pf.sum(1).squeeze(1) # NxFxT --> NxT
    Cmagsq_Pf = Cmagsq_Pf.sum(1).squeeze(1) # NxFxT --> NxT

    # Get SDR per samples
    inner_prod_sq = Cmag_Pf*Cmag_Pf # NxT
    numerator = inner_prod_sq # NxT
    #denominator = Pf_sum*Cmagsq_Pf - inner_prod_sq
    denominator = Cmagsq_Pf - inner_prod_sq  #NxT

    # Mean across time
    numerator = numerator.sum(1)/Tlist
    denominator = denominator.sum(1)/Tlist

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #sInvSDR_mean = torch.mean(sInvSDR).item()
    #print(sInvSDR_mean)

    #print(sInvSDR)

    return sInvSDR  #minibatchx1


def srcIndepSDR_freqpower_by_enhanced(out_real, out_imag, Tlist, Pf, eps=1e-10):  # scale invariant SDR loss function
    # out_real: NxFxT
    # out_imag: NxFxT
    # Pf: Fx1
    # Tlist: Nx1

    # power per freq
    #Pf = weights_per_freq*weights_per_freq # F
    #pdb.set_trace()
    #Pf_sum = Pf.sum() # = 1
    Wf_unsqueeze = Pf.sqrt().unsqueeze(0).unsqueeze(2) # 1xFx1



    enhanced_pow = out_real*out_real + out_imag*out_imag + eps
    enhanced_energy = torch.sqrt(enhanced_pow)

    Tlist = Tlist.float().cuda()

    energy_Wf = enhanced_energy*Wf_unsqueeze.expand_as(enhanced_energy) # NxFxT
    inner_prod_sq = torch.sum(torch.sum(energy_Wf, 2), 1) # Nx1
    enhanced_pow_sum = torch.sum(torch.sum(enhanced_pow, 2), 1) # Nx1
    pow_prod = Tlist*enhanced_pow_sum

    numerator = inner_prod_sq
    denominator = pow_prod - inner_prod_sq

    sInvSDR = 10*(torch.log10(numerator + eps) - torch.log10(denominator + eps))

    # sio.savemat('SI-SDR.mat', {'Wf':Wf_unsqueeze.data.cpu().numpy(),
    #                            'enhanced_pow':enhanced_pow.data.cpu().numpy(),
    #                            'Tlist':Tlist.data.cpu().numpy(),
    #                            'energy_Wf':energy_Wf.data.cpu().numpy(),
    #                            'inner_prod_sq':inner_prod_sq.data.cpu().numpy(),
    #                            'enhanced_pow_sum':enhanced_pow_sum.data.cpu().numpy(),
    #                            'pow_prod':pow_prod.data.cpu().numpy(),
    #                            'sInvSDR': sInvSDR.data.cpu().numpy()})

    return sInvSDR  #minibatchx1

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

'''
def sInvSDR_mag_given_output_pow(clean_real, clean_imag, output_pow, eps=1e-14):  # scale invariant SDR loss function
    # add eps for gradient explosion at sqrt(x)=0

    clean_pow = clean_real*clean_real + clean_imag*clean_imag + eps # NxFxT

    clean_pow_L2 = torch.sum(torch.sum(clean_pow, dim=2), dim=1)
    output_pow_L2 = torch.sum(torch.sum(output_pow, dim=2), dim=1)
    inner_product = torch.sum(torch.sum(torch.sqrt(clean_pow)*torch.sqrt(output_pow), dim=2), dim=1) # Re(x^Hy)

    power = clean_pow_L2*output_pow_L2
    inner_prod_sq = inner_product*inner_product

    numerator = inner_prod_sq
    denominator = power - inner_prod_sq

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #return torch.mean(wSDR) # scalar
    return sInvSDR # #minibatchx1
'''


def negative_MSE(clean_real, clean_imag, out_real, out_imag):

    power_diff = torch.pow(clean_real-out_real, 2) + torch.pow(clean_imag-out_imag, 2)
    negative_MSE = -torch.mean(torch.mean(power_diff, dim=2), dim=1)

    return negative_MSE # #minibatchx1


def cossim_time(clean, clean_est, eps=1e-10): # scale invariant SDR loss function
    bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
    def mSDRLoss(orig, est):
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return correlation / (energies + eps)

    cossim = mSDRLoss(clean, clean_est)
    #return torch.mean(wSDR) # scalar
    return cossim # #minibatchx1


def cossim_spec(clean_real, clean_imag, output_real, output_imag, eps=1e-10):
    # scale invariant SDR loss function, where clean & clean_est is both complex values

    inner_product = torch.sum(torch.sum(clean_real*output_real+clean_imag*output_imag, dim=2), dim=1) # Re(x^Hy)
    magnitude_clean = torch.sqrt(torch.sum(torch.sum(clean_real*clean_real + clean_imag*clean_imag, dim=2), dim=1))
    magnitude_output = torch.sqrt(torch.sum(torch.sum(output_real*output_real + output_imag*output_imag, dim=2), dim=1))

    cossim = inner_product/(magnitude_clean*magnitude_output + eps)

    return cossim # minibatchx1

#def cossim_mag(clean_mag, output_mag, eps=1e-10):
def cossim_mag(clean_real, clean_imag, output_real, output_imag, eps=1e-10):

    # scale invariant SDR loss function, where clean & clean_est is both magnitude
    # avoid sqrt() for gradient explosion at x=0

    # ver 4. original definition of cossim & add small eps on both vectors
    clean_pow = clean_real*clean_real + clean_imag*clean_imag + eps # NxFxT
    output_pow = output_real*output_real + output_imag*output_imag + eps # NxFxT

    clean_mag_L2 = torch.sqrt(torch.sum(torch.sum(clean_pow, dim=2), dim=1))
    output_mag_L2 = torch.sqrt(torch.sum(torch.sum(output_pow, dim=2), dim=1))
    inner_product = torch.sum(torch.sum(torch.sqrt(clean_pow)*torch.sqrt(output_pow), dim=2), dim=1)
    energies = clean_mag_L2*output_mag_L2
    cossim = inner_product/(energies + eps) # ver 4

    '''
    clean_sq = clean_real*clean_real + clean_imag*clean_imag
    output_sq = output_real*output_real + output_imag*output_imag
    inner_product = torch.sum(torch.sum(clean_sq*output_sq, dim=2), dim=1)
    clean_sq4 = torch.sum(torch.sum(clean_sq * clean_sq, dim=2), dim=1)
    output_sq4 = torch.sum(torch.sum(output_sq * output_sq, dim=2), dim=1)
    clean_out_sq4 = clean_sq4*output_sq4
    cossim = inner_product/(clean_out_sq4 + eps) # ver 3 (avoid using sqrt())
    '''

    #cossim = inner_product/(energies + eps) # ver 1
    #cossim = inner_product/(powers + eps) # ver 2

    return cossim # minibatchx1
