#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
from numpy.fft import fft,ifft

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    pi = np.pi
    cos = np.cos
    
    # number of block samples
    N = a + b
    
    # phase factor n0
    n0 = (b + 1)/2
    
    k = np.arange(0,int(N/2),1)
    n = np.arange(0,N,1)
    
    # Inverse MDCT slow
    if isInverse:
        x_n = np.zeros(int(N))
        for i in n:
            x_n[i] = np.sum(data*cos((2*pi/N)*(i + n0)*(k + (1/2))))
        
        # multiply by 2
        x_n *= 2
        return x_n
    
    # MDCT slow
    else:

        X_k = np.zeros(int(N/2))
        for i in k:
            X_k[i] = np.sum(data*cos((2*pi/N)*(n + n0)*(i + (1/2))))
            
        
        # multiply by 2/N
        X_k *= (2/N)
        return X_k


    ### YOUR CODE ENDS HERE ###

### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    pi = np.pi
    cos = np.cos
    exp = np.exp
    
    # number of block samples
    N = a + b
    halfN = N//2
    
    # phase factor n0
    n0 = (b + 1.)/2
    
    # forward transform
    if not isInverse:
        
        # pre-twiddle
        pre_twiddle = np.arange(N, dtype = np.float64)
        phase = -1j*pi/N
        pre_twiddle = data*exp(phase*pre_twiddle)
        
        # post-twiddle
        post_twiddle = np.linspace(0.5,halfN -0.5,halfN) 
        phase = -2j*pi*n0/N
        post_twiddle = np.exp(phase*post_twiddle)*2./N # add factor of 2/N
        
        # take fft
        return (post_twiddle*fft(pre_twiddle)[:halfN]).real 
#         x_fft_real = fft(pre_twiddle)[:halfN].real
#         X = post_twiddle*x_fft_real
        
    
    # inverse transform
    else:
        
        # pre-twiddle
        pre_twiddle = np.arange(N,dtype = np.float64) 
        phase = 2j*pi*n0/N
        pre_twiddle = exp(phase*pre_twiddle)
        
        # post-twiddle
        post_twiddle = np.linspace(n0,N+n0-1,N) 
        phase = 1j*pi/N
        post_twiddle = N*exp(phase*post_twiddle)
        return ( post_twiddle * ifft(pre_twiddle*np.concatenate((data,-data[::-1])))). real
#         X_ifft_real = ifft(pre_twiddle*np.concatenate((data,-data[::-1]))).real
#         x = post_twiddle*X_ifft_real


def IMDCT(data,a,b):

    return MDCT(data,a,b,isInverse=True)


#-----------------------------------------------------------------------------

