import torch
import math
import scipy.io as sio
import pdb

def Wdiff_realimag_gtnormalized(Wgt_real, Wgt_imag, West_real, West_imag, eps=1e-20):
    Wgt = torch.cat((Wgt_real, Wgt_imag), dim=1) # concat along mic dimension
    West = torch.cat((West_real, West_imag), dim=1) # concat along mic dimension

    err = Wgt - West # NxMxFxT

    Wgt_sq_sum = torch.sum(torch.sum(torch.sum(Wgt*Wgt, dim=3), dim=2), dim=1)
    distortion_to_signal_power = torch.sum(torch.sum(torch.sum(err*err, dim=3), dim=2), dim=1)/Wgt_sq_sum
    # note that distortion_to_signal_power is already normalized by #frame

    return -distortion_to_signal_power# x(-1) for -loss convention

def Wdiff_realimag(Wgt_real, Wgt_imag, West_real, West_imag,):
    F = Wgt_real.size(2)

    Wgt = torch.cat((Wgt_real, Wgt_imag), dim=1) # concat along mic dimension
    West = torch.cat((West_real, West_imag), dim=1) # concat along mic dimension

    err = Wgt - West # NxMxFxT
    #print('||Wgt|| = ' + str(Wgt.norm().item()))

    MSE = torch.sum(torch.sum(torch.sum(err*err, dim=3), dim=2), dim=1)/F

    return -MSE # x(-1) for -loss convention

def SDR_Wdiff_realimag(Wgt_real, Wgt_imag, West_real, West_imag, match_domain='realimag', eps=1e-20):
    # Tlist is for dummy

    Wgt = torch.cat((Wgt_real, Wgt_imag), dim=1) # concat along mic dimension
    West = torch.cat((West_real, West_imag), dim=1) # concat along mic dimension

    distortion = Wgt - West # NxMxFxT

    signal_power = torch.sum(torch.sum(torch.sum(Wgt*Wgt, dim=3), dim=2), dim=1)
    distortion_power = torch.sum(torch.sum(torch.sum(distortion*distortion, dim=3), dim=2), dim=1)

    SDR = 10*(torch.log10(signal_power+eps)-torch.log10(distortion_power+eps))
    # note that SDR is already normalized by #frame

    return SDR

def diff(enh_real, enh_imag, target_real, target_imag, Tlist, match_domain = 'realimag', eps=1e-16):
    if(match_domain == 'mag'):
        enh_mag = torch.sqrt(enh_real*enh_real + enh_imag*enh_imag + eps)
        target_mag = torch.sqrt(target_real*target_real + target_imag*target_imag + eps)
        err_mag = enh_mag - target_mag
        err_power = err_mag*err_mag
    elif(match_domain == 'realimag'):
        err_real = enh_real - target_real
        err_imag = enh_imag - target_imag
        err_power = err_real * err_real + err_imag * err_imag

    #err_power = torch.sum(torch.sum(err_real*err_real + err_imag*err_imag, dim=2), dim=1)
    negative_err_power = -torch.sum(torch.sum(err_power, dim=2), dim=1) # -sign for loss convention

    if(Tlist is not None): # source-dependent case
        negative_err_power = negative_err_power/(Tlist.float().cuda())

    #return err_power
    return negative_err_power


def WH_sum_diff(H_real, H_imag, W_real, W_imag, target_real, target_imag, Tlist, match_domain = 'realimag', eps=1e-16):

    # H, W: NxMxFxT
    WS_real = torch.sum(H_real*W_real-H_imag*W_imag, dim=1) # NxMxFxT --> NxFxT
    WS_imag = torch.sum(H_real*W_imag+H_imag*W_real, dim=1) # NxMxFxT --> NxFxT

    if(match_domain == 'mag'):
        WS_mag = torch.sqrt(WS_real*WS_real + WS_imag*WS_imag + eps)
        if(torch.is_tensor(target_real) and torch.is_tensor(target_imag)): # tensor
            target_mag = torch.sqrt(target_real*target_real + target_imag*target_imag + eps)
        else: # scalar
            target_mag = math.sqrt(target_real*target_real + target_imag*target_imag)
        err_mag = WS_mag - target_mag
        err_power = err_mag*err_mag
    elif(match_domain == 'realimag'):
        err_real = WS_real - target_real
        err_imag = WS_imag - target_imag
        err_power = err_real * err_real + err_imag * err_imag

    #err_power = torch.sum(torch.sum(err_real*err_real + err_imag*err_imag, dim=2), dim=1)
    negative_err_power = -torch.sum(torch.sum(err_power, dim=2), dim=1) # -sign for loss convention

    if(Tlist is not None): # source-dependent case
        negative_err_power = negative_err_power/(Tlist.float().cuda())

    #return err_power
    return negative_err_power

def distortion_em_mag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-12):
    F = clean_real.size(1)

    clean_mag = torch.sqrt(clean_real*clean_real + clean_imag*clean_imag+eps)
    output_mag = torch.sqrt(output_real*output_real + output_imag*output_imag+eps)

    distortion_mag =  output_mag-clean_mag
    distortion_power = torch.sum(torch.sum(distortion_mag*distortion_mag, dim=2), dim=1)

    Tlist_float = Tlist.float().cuda()

    distortion_power_frame_normalized_negative = -distortion_power/(Tlist_float*F) # x(-1) for -loss convention

    return distortion_power_frame_normalized_negative

def SDR_em_mag(clean_real, clean_imag, output_real, output_imag, Tlist, eps=1e-16):
    # Tlist as dummy variable
    clean_mag = torch.sqrt(clean_real*clean_real + clean_imag*clean_imag+eps)
    output_mag = torch.sqrt(output_real*output_real + output_imag*output_imag+eps)
    signal_power = torch.sum(torch.sum(clean_mag*clean_mag, dim=2), dim=1)

    distortion_mag = output_mag-clean_mag
    distortion_power = torch.sum(torch.sum(distortion_mag*distortion_mag, dim=2), dim=1)

    SDR = 10*(torch.log10(signal_power+eps) - torch.log10(distortion_power+eps))

    return SDR # #minibatchx1


def SDR_C_mag(clean_real, clean_imag, out_real, out_imag, Tlist, eps=1e-16):
    # Tlist as dummy variable
    N, F, Tmax = clean_real.size()
    #pdb.set_trace()
    cleanSTFT_pow = clean_real * clean_real + clean_imag * clean_imag + eps
    Cr = (out_real * clean_real + out_imag * clean_imag) / cleanSTFT_pow
    Ci = (-out_real * clean_imag + out_imag * clean_real) / cleanSTFT_pow

    Cmag = torch.sqrt(Cr * Cr + Ci * Ci + eps)
    #sio.savemat('Cmag.mat', {'cleanSTFT_pow':cleanSTFT_pow.data.cpu().numpy(), 'Cr':Cr.data.cpu().numpy(), 'Ci':Ci.data.cpu().numpy(), 'Cmag_py':Cmag.data.cpu().numpy()})

    distortion_mag = Cmag-1
    # make distortion 0for garbage frames
    for i, l in enumerate(Tlist):  # zero padding to output audio
        distortion_mag[i, :, l:] = 0

    Pdistortion = torch.sum(torch.sum(distortion_mag*distortion_mag, dim=2), dim=1)

    Psignal = Tlist.float().cuda()*F
    #print('Psignal = ')
    #print(Psignal)

    #print('Pdistortion = ')
    #print(Pdistortion)

    SDR = 10*(torch.log10(Psignal+eps) - torch.log10(Pdistortion+eps))

    return SDR # #minibatchx1


def reference_position_demixing(X_real, X_imag, W_real, W_imag, Tlist):
    F = X_real.size(2)

    # size: NxMxFxT
    XW_real = torch.sum(W_real*X_real-W_imag*X_imag, dim=1) # NxFxT
    XW_imag = torch.sum(W_real*X_imag+W_imag*X_real, dim=1) # NxFxT

    XW_power = torch.sum(torch.sum(XW_real*XW_real + XW_imag*XW_imag, dim=2), dim=1) # Nx1

    Tlist_float = Tlist.float().cuda()
    XW_power_frame_normalized_negative = -XW_power/(Tlist_float*F) # x(-1) for -loss convention

    return XW_power_frame_normalized_negative