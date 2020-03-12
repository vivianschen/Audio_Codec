#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
window.py -- Defines functions to window an array of data samples
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import mpmath
import math
import matplotlib.pyplot as plt

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    
    sin = np.sin
    pi = np.pi
    
    # number of samples
    N = len(dataSampleArray)
    # array of samples
    n = np.arange(N)
    # sin window
    w_s = sin(pi*(n+0.5)/N)
    
    # windowed copy of input array
    dataSampleArrayWind = w_s*dataSampleArray
    
    return dataSampleArrayWind
    

    ### YOUR CODE ENDS HERE ###


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    cos = np.cos
    pi = np.pi
    
    # number of samples
    N = len(dataSampleArray)
    # array of samples
    n = np.arange(N)
    # hanning window
    w_h = (1/2)*(1-cos(2*pi*(n+1/2)/N))
    
    # windowed copy of input array
    dataSampleArrayWind = w_h*dataSampleArray
    
    return dataSampleArrayWind
    

    
    ### YOUR CODE ENDS HERE ###


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following the KDB Window handout in the 
	Canvas Files/Assignments/HW3 folder
    """

    ### YOUR CODE STARTS HERE ###
    
     ### YOUR CODE STARTS HERE ###
    cos = np.cos
    pi = np.pi
    sqrt = np.sqrt
    
    # number of samples
    N = len(dataSampleArray)
    # array of samples
    n = np.arange(N)
    # alpha 
    alpha = 4
    
    # KBD window
    w_kb = np.i0(pi*alpha*sqrt(1-(((2*n+1)/N)-1)**2))/np.i0(pi*alpha)
    
    # windowed copy of input array
    dataSampleArrayWind = w_kb*dataSampleArray
    
    return dataSampleArrayWind


    ### YOUR CODE ENDS HERE ###
    



#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    
    cos = np.cos
    pi = np.pi
    
    n = np.arange(1024)
    fs = 44100
    x = cos(2*pi*3000*n/fs)
    N = len(x)
    a = N/2
    b = N/2
    
    # sin windowed x, fft and MDCT
    sin_windowed_x = SineWindow(x)
    sin_windowed_x_fft = np.fft.fft(sin_windowed_x)
    sin_windowed_x_MDCT = MDCT(sin_windowed_x,a,b)
    
    print(sin_windowed_x)
    
    #_wsfft_ = (1/N**2)*np.sum(np.abs(sin_windowed_x_fft)**2)
    #_wsmdct_ = (1/N**2)*np.sum(np.abs(sin_windowed_x_MDCT)**2)

    sin_windowed_x_fft_SPL = 96 + 10 * np.log((4*2 / (N**2)) * np.abs(sin_windowed_x_fft)**2)
    sin_windowed_x_MDCT_SPL = 96 + 10 * np.log((8*2 / (N**2))* np.abs(sin_windowed_x_MDCT)**2)
    
    # hanning windowed x and fft
    hann_windowed_x = HanningWindow(x)
    hann_windowed_x_fft = np.fft.fft(hann_windowed_x)
    
    #_wh_ = (1/N**2)*np.sum(np.abs(hann_windowed_x_fft)**2)
    
    hann_windowed_x_fft_SPL = 96 + 10 * np.log((4*(8/3) / (N**2)) * np.abs(hann_windowed_x_fft)**2)
    
    
    # KBD window
    kbd_windowed_x = KBDWindow(x)
    kbd_windowed_x_fft = np.fft.fft(kbd_windowed_x)
    
    _wkbd_ = (1/N**2)*np.sum(np.abs(kbd_windowed_x_fft)**2)
    
    kbd_windowed_x_fft_SPL = 96 + 10 * np.log((4 / ((N**2) * _wkbd_)) * np.abs(kbd_windowed_x_fft)**2)
    
    # plot time sine window vs hanning window in time domain
    plt.figure()
    plt.plot(sin_windowed_x, label = 'Sine Window')
    plt.plot(hann_windowed_x, label = 'Hanning Window')
    plt.plot(kbd_windowed_x, label = 'KBD Window')
    plt.grid()
    plt.title('Sine Window vs. Hanning Window')
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # plot time sine window vs hanning window in frequency domain
    x_axis = np.arange(0,int(N/2))*(fs/N)
    plt.figure()
    plt.plot(x_axis, sin_windowed_x_fft_SPL[0:int(N/2)], label = 'Sine Window FFT')
    plt.plot(x_axis, sin_windowed_x_MDCT_SPL[0:int(N/2)], label = 'Sine Window MDCT')
    plt.plot(x_axis, hann_windowed_x_fft_SPL[0:int(N/2)], label = 'Hanning Window FFT')
    plt.plot(x_axis, kbd_windowed_x_fft_SPL[0:int(N/2)], label = 'KBD Window FFT')
    plt.grid()
    plt.title('Sine Window vs. Hanning Window')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL')
    plt.legend()
    plt.show()
    
    # plot time sine window vs hanning window in frequency domain (ZOOM-IN)
    x_axis = np.arange(0,int(N/2))*(fs/N)
    plt.figure()
    plt.plot(x_axis, sin_windowed_x_fft_SPL[0:int(N/2)], label = 'Sine Window FFT')
    plt.plot(x_axis, sin_windowed_x_MDCT_SPL[0:int(N/2)], label = 'Sine Window MDCT')
    plt.plot(x_axis, hann_windowed_x_fft_SPL[0:int(N/2)], label = 'Hanning Window FFT')
    plt.plot(x_axis, kbd_windowed_x_fft_SPL[0:int(N/2)], label = 'KBD Window FFT')
    plt.grid()
    plt.title('Sine Window vs. Hanning Window')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL')
    plt.legend()
    plt.ylim([40,100])
    plt.xlim([2500,3500])
    plt.show()

    pass # THIS DOES NOTHING

    ### YOUR TESTING CODE ENDS HERE ###


