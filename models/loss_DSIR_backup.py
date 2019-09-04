import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sInvSDR_time(clean, clean_est, eps=1e-10): # scale invariant SDR loss function
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
    correlation = bsum(clean * clean_est)
    energies = torch.norm(clean, p=2, dim=1) * torch.norm(clean_est, p=2, dim=1)

    numerator = correlation*correlation
    denominator = energies*energies - correlation*correlation

    sInvSDR = 10*torch.log10(numerator/(denominator + eps))

    return sInvSDR # #minibatchx1

def sInvSDR_spec(clean_real, clean_imag, output_real, output_imag, eps=1e-10): # scale invariant SDR loss function

    inner_product = torch.sum(torch.sum(clean_real*output_real + clean_imag*output_imag, dim=2), dim=1) # Re(x^Hy)
    power_clean = torch.sum(torch.sum(clean_real*clean_real + clean_imag*clean_imag, dim=2), dim=1)
    power_output = torch.sum(torch.sum(output_real*output_real + output_imag*output_imag, dim=2), dim=1)

    inner_product_sq = inner_product*inner_product
    power_clean_output = power_clean*power_output

    numerator = inner_product_sq
    denominator = power_clean_output - inner_product_sq

    sInvSDR = 10*(torch.log10(numerator+eps) - torch.log10(denominator + eps))

    #return torch.mean(wSDR) # scalar
    return sInvSDR # #minibatchx1

def var_time(W):
    # time dimension in W: 3
    var = torch.var(W, dim=3).mean()

    return var


#def sInvSDR_mag(clean_mag, output_mag, eps=1e-10): # scale invariant SDR loss function
def sInvSDR_mag(clean_real, clean_imag, output_real, output_imag, eps=1e-10):  # scale invariant SDR loss function
    # add eps for gradient explosion at sqrt(x)=0

    clean_pow = clean_real*clean_real + clean_imag*clean_imag + eps # NxFxT
    output_pow = output_real*output_real + output_imag*output_imag + eps # NxFxT

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