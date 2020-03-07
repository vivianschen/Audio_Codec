"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc, BitAllocUniform, BitAllocConstSNR, BitAllocConstNMR  #allocates bits to scale factor bands given SMRs
from midSide import *


def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1.*(1<<overallScaleFactor)
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN,dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines =codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level


    # IMDCT and window the data for this channel
    data = SineWindow( IMDCT(mdctLine, halfN, halfN) )  # takes in halfN MDCT coeffs

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    
    
    # Check if we want MID SIDE Encoding or not
    codingParams.use_MS = MidSideCorrelation(data[0], data[1], codingParams)
            
    if codingParams.use_MS == 0:    
        # loop over L and R channels and separately encode each one
        for iCh in range(codingParams.nChannels):
            (s,b,m,o,codingParams.bitReservoir) = EncodeSingleChannel(data[iCh],codingParams)
            scaleFactor.append(s)
            bitAlloc.append(b)
            mantissa.append(m)
            overallScaleFactor.append(o)
        
            # print("not using MS")
        
    else:
        
        #  print("using MS")
        
        halfN = codingParams.nMDCTLines

        # window data and compute MDCT for left and right channels
        timeSamplesLeft = data[0]
        mdctTimeSamplesLeft = SineWindow(data[0])
        mdctLinesLeft = MDCT(mdctTimeSamplesLeft, halfN, halfN)[:halfN]
        
        timeSamplesRight = data[1]
        mdctTimeSamplesRight = SineWindow(data[1])
        mdctLinesRight = MDCT(mdctTimeSamplesRight, halfN, halfN)[:halfN]

        
        # compute MDCT and time domain mid and side channels 
        mdctLinesMid = (mdctLinesLeft + mdctLinesRight)/np.sqrt(2) 
        timeSamplesMid = (timeSamplesLeft + timeSamplesRight)/np.sqrt(2) 

        mdctLinesSide = (mdctLinesLeft - mdctLinesRight)/np.sqrt(2) 
        timeSamplesSide = (timeSamplesLeft - timeSamplesRight)/np.sqrt(2) 


        # compute overall scale factor for this block and boost mdctLines using it
        maxLineMid = np.max( np.abs(mdctLinesMid) )
        overallScaleMid = ScaleFactor(maxLineMid,codingParams.nScaleBits)  #leading zeroes don't depend on nMantBits
        mdctLinesMid *= (1<<overallScaleMid)

        maxLineSide = np.max( np.abs(mdctLinesSide) )
        overallScaleSide = ScaleFactor(maxLineSide,codingParams.nScaleBits)  #leading zeroes don't depend on nMantBits
        mdctLinesSide *= (1<<overallScaleSide)

        
        # compute SMRs for mid and side channels
        SMRs_Mid = CalcSMRs(timeSamplesMid, mdctLinesMid, overallScaleMid, codingParams.sampleRate, codingParams.sfBands)
        SMRs_Side = CalcSMRs(timeSamplesSide, mdctLinesSide, overallScaleSide, codingParams.sampleRate, codingParams.sfBands)

        # check which SMR is larger (mid or side) for each band and subtract 6 dB 
        for i in range(len(SMRs_Mid)):
            
            if SMRs_Mid[i] > SMRs_Side[i]:
                SMRs_Mid[i] -= 6.02
            else:
                SMRs_Side[i] -= 6.02
            

        # encode mid channel
        (s,b,m,o,codingParams.bitReservoir) = EncodeMidSideChannel(data[0], data[1], 0, mdctLinesMid, SMRs_Mid, overallScaleMid, codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
        
        # encode side channel
        (s,b,m,o,codingParams.bitReservoir) = EncodeMidSideChannel(data[0], data[1], 1, mdctLinesSide, SMRs_Side, overallScaleSide, codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
   
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams.bitReservoir, codingParams.use_MS)


def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    
    bitBudget += codingParams.bitReservoir # add bit reservoir bits to bit budget
    
#     print("init bitBudget", bitBudget);


    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    mdctTimeSamples = SineWindow(data)
    mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    
    # perform bit allocation using SMR results
    bitAlloc, codingParams.bitReservoir = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs, codingParams.bitReservoir)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale, codingParams.bitReservoir)

def EncodeMidSideChannel(dataLeft, dataRight, midOrSide, mdctLines, SMRs, overallScale, codingParams):

    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    bitBudget -= 1 # less the MS flag bit
    
    bitBudget += codingParams.bitReservoir # add bit reservoir bits to bit budget

    
    # perform bit allocation using SMR results
    bitAlloc, codingParams.bitReservoir = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs, codingParams.bitReservoir)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale, codingParams.bitReservoir)  
            
            
            