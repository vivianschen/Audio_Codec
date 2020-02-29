#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from window import *
from mdct import *

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity 
    """
    
    zero_ = 1*(10**(-15))
    
    spl = 96 + 10*np.log10(intensity + zero_)
    spl = np.where(spl <= -30, -30, spl)
     
    return spl 

def Intensity(spl):
    """
    Returns the intensity  for SPL spl
    """

    I = np.where(spl < -30, 0, 10**((spl-96)/10))  
    
    return I

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    
    f = np.where(f < 10, 10, f)
    A = 3.64*np.power(f / 1000, -0.8) - 6.5*np.exp(-0.6*np.power((f / 1000) - 3.3,2)) + (10**-3)*(f / 1000)**4
    thresh = A      
    
    return thresh

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    arctan = np.arctan
    bark = 13*arctan(0.76*(f / 1000)) + 3.5*arctan(np.power(f / 7500, 2))
    return bark

class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self,f,SPL,isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        self.f = f
        self.z = Bark(f)
        self.spl = SPL
        self.isTonal = isTonal
        if self.isTonal:
            self.drop = 16
        else:
            self.drop = 6

    def IntensityAtFreq(self,freq):
        """The intensity at frequency freq"""
        I = Masker.IntensityAtBark(self, Bark(freq))
        return I 
        

    def IntensityAtBark(self,z):
        """The intensity at Bark location z"""
    
        
        dz = z - self.z
        if dz < (-1/2):
            spreading = -27*(np.abs(dz)-(1/2))
        elif dz > (1/2):
            spreading = -27 + 0.367*np.max(self.spl - 40,0)*(np.abs(dz)-(1/2))
        else:
            spreading = self.spl - self.drop

        #spread = (SPL(10**(spread/10))) - 96 + self.spl - drop

        I = Intensity(maskedDB)
        
        return I
        

    def vIntensityAtBark(self,zVec):
        """The intensity at vector of Bark locations zVec"""
        
        dzVec = zVec - self.z
        
        spreadVec = (-27*(np.abs(dzVec) - 0.5))*(dzVec < -0.5) + (-27 + 0.367*np.max(self.spl - 40,0))*(np.abs(dzVec) - 0.5)*(dzVec > 0.5)

        spreadVec = spreadVec + self.spl - self.drop #SPL(10**(spreadVec/10)) - 96 + self.spl - drop

        IVec = Intensity(spreadVec)          
        return IVec


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 24000]  

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    
    freq_factor = 0.5*sampleRate/nMDCTLines
    
    # create array of sampels lenght of MDCTLines
    n = np.arange(0,nMDCTLines)
    
    # convert samples to frequency bins
    freqs = (n + 0.5)*freq_factor
    
    # array to store number of frequency bins/lines per scale factor band
    num_lines = np.zeros((len(flimit)), dtype = np.int)
    
    # counter variable
    count = 0
    # indx of flimit array
    indx = 0
    # count number of frequency bins in each scale factor band
    for i in range(len(freqs)):
        if indx < len(flimit):
            if freqs[i] <= flimit[indx]:
                count += 1
            else:
                indx += 1
                count = 0
                if indx < len(flimit):
                    if freqs[i] <= flimit[indx]:
                        count += 1 
            num_lines[indx] = count 
    
    return num_lines

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
     
        # total number of scale factor bands
        self.nBands = len(nLines) 
        # lower line and upper line
        self.lowerLine = np.zeros((len(nLines)), dtype = np.int)
        self.upperLine = np.zeros((len(nLines)), dtype = np.int)
        
        for i in range(len(nLines)):
           
            # set lowerLine
            if i == 0:
                self.lowerLine[i] = 0
            elif i < len(nLines):
                self.lowerLine[i] = self.upperLine[i - 1] + 1
            
            # set upperLine
            self.upperLine[i] = self.lowerLine[i] + nLines[i] - 1
        
        nLines = np.asarray(nLines)
        self.nLines = nLines #self.upperLine - self.lowerLine + 1  

def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    fs = sampleRate
    N = len(data)
    
    # window data and comput data_fft
    w_h = HanningWindow(data)
    data_fft = np.fft.fft(w_h)

    # calculate intensity of data_fft
    intensity_fft = (4/((N**2)*(3/8)))*np.power(np.abs(data_fft),2)

    # calculate SPL
    spl = SPL(intensity_fft)

    # get the fft frequencies 
    freq = np.fft.fftfreq(len(spl),1/fs) + fs/(2*N)

    # get only positive half of fft and freq vectors
    spl_pos = spl[:int(len(spl)/2)]
    freq_pos = freq[:int((len(spl)/2))]

    # find the peaks of spl signal
    maxima_index = []
    for j in range(1, len(spl_pos)-1):
        if (j == len(spl_pos)-1):
            next_bin = 0

        else:
            prev_bin = spl_pos[j-1]
            next_bin = spl_pos[j+1]

        if (prev_bin < spl_pos[j]) & (next_bin < spl_pos[j]):
            maxima_index.append(j)

    np.asarray(maxima_index)

    spl_mask = np.zeros((len(maxima_index)))
    freq_mask = np.zeros((len(maxima_index)))
    indx = 0

    # calculate the weighted average of the intensity for each peak 
    for p in maxima_index:
        # calculate intensity at each peak, and at the negihboring beans 
        i_left = Intensity(spl_pos[p-1])
        i_center = Intensity(spl_pos[p])
        i_right = Intensity(spl_pos[p+1])

        # compute intensity of the peak
        I = i_left + i_center + i_right
        
        # caclulate SPL of peak
        spl_mask[indx] = SPL(I)
        
        # calculate frequency of peak 
        freq_mask[indx] = (i_left*freq_pos[p-1] + i_center*freq_pos[p] + i_right*freq_pos[p+1])/I
        indx += 1


    # create massker for each peak 
    maskers = []
    for i in range(len(freq_mask)):

        # create masker object
        masker = Masker(freq_mask[i],spl_mask[i],True)
        maskers.append(masker)   

    maskers = np.asarray(maskers)

    # create threshold curve with all frequencies as a starter mask
    
    max_mask_db = Thresh(freq_pos)

    i = 0
    for m in maskers:   
        # compute spl masker curves
        spl_masker_db = SPL(m.vIntensityAtBark(Bark(freq_pos)))
        i += 1

        # compute overall max mask
        max_mask_db = np.maximum(max_mask_db, spl_masker_db)

    return max_mask_db
    

def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency coefficients for the time domain samples
                            in data; note that the MDCT coefficients have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  corresponds to an overall scale factor 2^MDCTscale for the set of MDCT
                            frequency coefficients
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Combines their relative masking curves and the hearing threshold
                to calculate the overall masked threshold at the MDCT frequency locations. 
				Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    
    # signal to maks: signal - mask for ecah line
    # smr=  np.max(signal-mask)
    # from all the lines(frequency bins) find the max smr
    # find max smr from each frequqncy bin for each sub band
    # will end up with 25 smrs in an array -> [smr1,srm2,...]
    
    nBands = sfBands.nBands
    lowerLine = sfBands.lowerLine
    upperLine = sfBands.upperLine
    
    N = len(data)
    fs = sampleRate
    
    # scale down MDCTdata by 2^MDCTscale
    MDCTdata = MDCTdata/(2**MDCTscale)
    
    # compute SPL of MDCTdata
    MDCTdata_spl = SPL(4*np.power(np.abs(MDCTdata),2)) 
    
    # compute mask treshold 
    mask_thresh_spl = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)
    
    SMR_bands = np.zeros((nBands))
    
    # calculate SMR of each MDCT line
    SMR_lines =  MDCTdata_spl - mask_thresh_spl
    
    for i in range(nBands):
        # take the max SMR of MDCT lines per band
        current_band_SMRs = SMR_lines[int(lowerLine[i]):int(upperLine[i])+1]
        SMR_bands[i] = np.max(current_band_SMRs)
    
#     plt.figure(figsize = (fig_w,fig_h))
#     MDCT_freqs = np.arange(0, fs/2, fs/N) + 0.5
#     mask_freqs = np.arange(0, fs/2, fs/N)
#     plt.semilogx(MDCT_freqs, MDCTdata_spl, label = "MDCT SPL"); 
#     plt.semilogx(mask_freqs, mask_thresh_spl, label = "Mask Threshold SPL"); 
#     plt.grid()
#     plt.title("MDCT SPL vs. Mask Threshold SPL")
#     plt.xlabel("Samples (n)")
#     plt.ylabel("SPL (dB)")
#     plt.ylim([-40, 100])
#     plt.legend()
#     plt.show()
    
    return SMR_bands
    
    
    

     

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    
    fig_w = 12
    fig_h = 6
    
    # 1.b) 

    pi = np.pi
    cos = np.cos
    A0 = 0.43; A1 = 0.24; A2 = 0.15; A3 = 0.09; A4 = 0.05; A5 = 0.04
    f0 = 440; f1 = 550; f2 = 660; f3 = 880; f4 = 4400; f5 = 8800
    fs = 48000
    N = np.array([512, 2048, 1024])

    for i in range(len(N)):

        n = np.arange(0,N[i])

        # x[n]
        x_n = A0*cos(2*pi*f0*n/fs) + A1*cos(2*pi*f1*n/fs) + A2*cos(2*pi*f2*n/fs) + A3*cos(2*pi*f3*n/fs) + A4*cos(2*pi*f4*n/fs) + A5*cos(2*pi*f5*n/fs)

        # window x[n] and comput X[k] 
        w_h = HanningWindow(x_n)
        X_k = np.fft.fft(w_h)

        # calculate intensity of X_k
        intensity_fft = (4/((N[i]**2)*(3/8)))*np.power(np.abs(X_k),2)

        # calculate SPL
        X_k_spl = SPL(intensity_fft)

        # get the fft frequencies 
        freq = np.fft.fftfreq(len(X_k_spl),1/fs)

        # get only positive half of fft and freq vectors
        spl_pos = X_k_spl[:int(len(X_k_spl)/2)]
        freq_pos = freq[:int((len(X_k_spl)/2))]

        # plot signal in time domain
        plt.figure(figsize = (fig_w,fig_h))
        plt.plot(n,x_n)
        plt.grid()
        plt.title("x[n] for N = " + str(N[i]))
        plt.xlabel("Samples (n)")
        plt.ylabel("Amplitude")
        plt.show()

        # plot spl of signal in frequency domain
        plt.figure(figsize = (fig_w,fig_h))
        plt.semilogx(freq_pos,spl_pos); 
        plt.grid()
        plt.title("SPL level of X[k] for N = " + str(N[i]))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.show()

        # find the peaks of spl signal
        maxima_index = []
        for j in range(1, len(spl_pos)-1):

            if (j == len(spl_pos)-1):
                next_bin = 0

            else:
                prev_bin = spl_pos[j-1]
                next_bin = spl_pos[j+1]

            if (prev_bin < spl_pos[j]) & (next_bin < spl_pos[j]):
                maxima_index.append(j)

        np.asarray(maxima_index)

        # maxima_indx = signal.argrelextrema(spl_pos,np.greater)


        spl_mask = np.zeros((len(maxima_index)))
        freq_mask = np.zeros((len(maxima_index)))
        indx = 0

        # calculate the weighted average of the intensity for each peak 
        for p in maxima_index:
            # calculate intensity at each peak, and at the negihboring beans 
            i_left = Intensity(spl_pos[p-1])
            i_center = Intensity(spl_pos[p])
            i_right = Intensity(spl_pos[p+1])

            # compute intensity of the peak
            I = i_left + i_center + i_right

            # caclulate SPL of peak
            spl_mask[indx] = SPL(I)

            # calculate frequency of peak 
            freq_mask[indx] = (i_left*freq_pos[p-1] + i_center*freq_pos[p] + i_right*freq_pos[p+1])/I
            indx += 1

        print("spl_mask:", spl_mask)
        print("mask frequencies for N = " + str(N[i]) + ": ", freq_mask)
        
        
    # 1.c)

    # create array of frequencies
    freqs = np.arange(20,20000,1)

    # calculate threshold in quiet
    thresh = Thresh(freqs)

    # plot threshold in quiet and SPL curve
    plt.figure(figsize = (fig_w,fig_h))
    plt.semilogx(freqs, thresh, label = "Threshold in Quiet")
    plt.semilogx(freq_pos, spl_pos, label = "SPL Curve of X[k] for N = 1024"); 
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SPL (dB)")
    plt.title("Threshold in Quiet vs. Signal SPL Curve")
    plt.legend()
    plt.show()

    
    # 1.d)

    # array of critical band lower frequencies f1
    f_vec = np.array([0,100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500])
    # convert to critical band lower frequencies to Bark values
    bark_vec = Bark(f_vec)
    print("Bark Values: ",bark_vec)


   # 1.e) 

    # create masksers for each peak 
    maskers = []
    for i in range(len(freq_mask)):

        # create masker object
        masker = Masker(freq_mask[i],spl_mask[i],True)
        maskers.append(masker)   

    maskers = np.asarray(maskers)

    # create threshold curve with all frequencies as a starter mask
    tresh_freq_pos = Thresh(freq_pos)
    max_mask = tresh_freq_pos

    plt.figure(figsize = (fig_w,fig_h))
    i = 0

    for m in maskers:   
        # compute spl masker curves
        spl_masker_db = SPL(m.vIntensityAtBark(Bark(freq_pos)))

        # plot spl masker curves
        plt.semilogx(freq_pos, spl_masker_db, label = "Mask Curve " + str(i))
        i += 1

        # compute overall mask
        max_mask = np.maximum(max_mask, spl_masker_db)

    # plot spl curve and threshold in quiet on top of masker curves
    plt.semilogx(freq_pos, spl_pos, label = "SPL Curve of X[k] for N = 1024")
    plt.semilogx(freq_pos, tresh_freq_pos, label = "Threshold in Quiet")
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SPL (dB)")
    plt.legend(loc='upper left')
    plt.title("SPL Signal Curve, Threshold in Quiet, and Mask Curves")
    plt.ylim([-20, 90])
    plt.show()

    # plot max masker curve with spl curve and theshold in quiet
    plt.figure(figsize = (fig_w,fig_h))
    plt.semilogx(freq_pos, spl_pos, label = "SPL Curve of X[k] for N = 1024")
    plt.semilogx(freq_pos, max_mask, label = "Max Mask Threshold Curve")
    plt.semilogx(freq_pos, tresh_freq_pos, label = "Threshold in Quiet")
    plt.grid()
    plt.title("SPL Signal Curve, Threshold in Quiet, and Max Mask Curve")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SPL (dB)")
    plt.ylim([-20, 90])
    plt.xlim([40,20000])


    # Plot critical bands boundaries
    for i in range(len(cbFreqLimits)):
        if i == 0:
            plt.axvline(x = cbFreqLimits[i], color = 'k', linestyle = '--', alpha = 0.6, label = "Critical Bands")

        else:
            plt.axvline(x = cbFreqLimits[i], color = 'k', linestyle = '--', alpha = 0.6)

    plt.legend()
    plt.show()
    
    # 1.f)
    nMDCTLines = 512
    fs = 48000
    nLines = AssignMDCTLinesFromFreqLimits(nMDCTLines, fs)
    sfBands = ScaleFactorBands(nLines)

    # 1.g)
    data = x_n

    # compute MDCT of sine windowed data
    N = 1024
    a = int(N/2)
    b = int(N/2)
    MDCTdata = MDCT(SineWindow(data),a,b); MDCTscale = 0

    sampleRate = fs
    sfBands = ScaleFactorBands(nLines)

    # compute maximum SRM in each scale factor band
    SMRs = CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands)
    print("SMRs", SMRs)



# In[ ]: