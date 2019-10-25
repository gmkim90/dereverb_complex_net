import torch
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

def SDR_Wdiff_realimag(Wgt_real, Wgt_imag, West_real, West_imag, eps=1e-20):
    # Tlist is for dummy

    Wgt = torch.cat((Wgt_real, Wgt_imag), dim=1) # concat along mic dimension
    West = torch.cat((West_real, West_imag), dim=1) # concat along mic dimension

    distortion = Wgt - West # NxMxFxT

    signal_power = torch.sum(torch.sum(torch.sum(Wgt*Wgt, dim=3), dim=2), dim=1)
    distortion_power = torch.sum(torch.sum(torch.sum(distortion*distortion, dim=3), dim=2), dim=1)

    SDR = 10*(torch.log10(signal_power+eps)-torch.log10(distortion_power+eps))
    # note that SDR is already normalized by #frame

    return SDR

def WH_sum_diff(H_real, H_imag, W_real, W_imag, target_real, target_imag):

    # H, W: NxMxFxT
    WS_real = torch.sum(H_real*W_real-H_imag*W_imag, dim=1) # NxFxT
    WS_imag = torch.sum(H_real*W_imag+H_imag*W_real, dim=1) # NxFxT

    err_real = WS_real - target_real
    err_imag = WS_imag - target_imag

    #err_power = torch.sum(torch.sum(err_real*err_real + err_imag*err_imag, dim=2), dim=1)
    negative_err_power = -torch.sum(torch.sum(err_real * err_real + err_imag * err_imag, dim=2), dim=1) # -sign for loss convention

    #return err_power
    return negative_err_power