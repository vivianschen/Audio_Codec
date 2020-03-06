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
    
    
#     # get lower and upper lines for each band
#     sfBands = codingParams.sfBands
#     nBands = sfBands.nBands
#     lowerLine = sfBands.lowerLine
#     upperLine = sfBands.upperLine
    
#     # initialize diff and summ arrays
#     diff = np.zeros((nBands))
#     summ = np.zeros((nBands))
    
#     # initialize use_MS array of bools
#     use_MS = np.zeros((nBands), dtype = np.bool)
    
#     # calculate correlation for each band
#     for i in range(nBands):
#         # get fft lines for current band
#         curr_band_fft_left = dataLeft_fft[int(lowerLine[i]):int(upperLine[i])+1]
#         curr_band_fft_right = dataRight_fft[int(lowerLine[i]):int(upperLine[i])+1]
        
#         # compute diff and sum between left and right bands
#         diff[i] = np.sum(abs(curr_band_fft_left**2 - curr_band_fft_right**2))
#         summ[i] = np.sum(abs(curr_band_fft_left**2 + curr_band_fft_right**2))
        
#         # check whether to use mid side coding or not
#         if diff < 0.8*summ:
#             use_MS[i] = True
#         elif summ < 0.8*diff:
#             use_MS[i] = True
#         else:
#             use_MS[i] = False
            
#         if use_MS[i]:
#             mid = 
#             side = 
        