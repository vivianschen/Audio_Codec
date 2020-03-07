import numpy as np
from psychoac import *



def MidSideCorrelation(dataLeft, dataRight, codingParams):

    # window data and comput data_fft
        
    # LEFT
    windLeft = HanningWindow(dataLeft)
    dataLeft_fft = np.fft.fft(windLeft)
    # RIGHT
    windRight = HanningWindow(dataRight)
    dataRight_fft = np.fft.fft(windRight)
    
    
    # calculate correlation of left and right channels
    diff = np.sum(abs(dataLeft_fft**2 - dataRight_fft**2))
    summ = np.sum(abs(dataLeft_fft**2 + dataRight_fft**2))
        
    # check whether to use mid side coding or not
    
    # use MS coding 
    if diff < 0.8*summ:
        use_MS = 1 
    # don't use MS coding
    else:
        use_MS = 0
        
    return use_MS
    
 
        