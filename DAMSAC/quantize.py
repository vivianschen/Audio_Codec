"""
quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np

### Problem 1.a.i ###
def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    #Notes:
    #The overload level of the quantizer should be 1.0

    ### YOUR CODE STARTS HERE ###
    
    # define sign bit
    if aNum >= 0:
        s = 0 << (nBits - 1)
    else:
        s = 1 << (nBits - 1)
        
        
    # define |code|
    if abs(aNum) >= 1:
        abs_code = 2**(nBits-1)-1
    else:
        abs_code = int(((2**(nBits)-1)*abs(aNum)+1)/2)
        
        
    # [s][|code|]
    aQuantizedNum = s | abs_code
    
    return aQuantizedNum


### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """
    
    ### YOUR CODE STARTS HERE ###
    # get s value
    s = aQuantizedNum >> nBits - 1
    
    # define sign value
    if s == 0:
        sign = 1
    elif s == 1:
        sign = -1
        
    # create mask 
    mask = ~(1 << (nBits - 1))
    # and mask with aQuantizedNum to get |code|
    abs_code = aQuantizedNum & mask
    
    # define |number|
    abs_number = 2*abs_code/(2**(nBits)-1)
    
    # define aNum
    aNum = sign*abs_number
    
    return aNum
    
    ### YOUR CODE ENDS HERE ###


### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """
    #Notes:
    #Make sure to vectorize properly your function as specified in the homework instructions

    ### YOUR CODE STARTS HERE ###
    
    if nBits <=0:
        return np.zeros((len(aNumVec)),dtype=np.uint64)
    
    # get s value
    sVec = np.where(aNumVec >= 0, 0 << nBits - 1, 1 << nBits - 1)
    
    
    # define |code| vector
    abs_code1 = 2**(nBits-1)-1
    abs_code0 = ((np.dot(2**(nBits)-1,np.abs(aNumVec))+1)/2).astype(np.int)
    abs_codeVec = np.where(np.abs(aNumVec) >=1, abs_code1, abs_code0)
    
    # [s][|code|]
    aQuantizedNumVec = sVec | abs_codeVec

    ### YOUR CODE ENDS HERE ###

    return aQuantizedNumVec



### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """

    ## YOUR CODE STARTS HERE ###
    
    # define sign value
    signVec = np.where(aQuantizedNumVec >> nBits - 1 == 0, 1, -1)
        
    # create mask 
    mask = ~(1 << nBits - 1)
    # and mask with aQuantizedNum to get |code|
    abs_codeVec = aQuantizedNumVec & mask
    
    # define |number|
    abs_numberVec = 2*abs_codeVec/(2**(nBits)-1)
    
    # define aNum
    aNumVec = signVec*abs_numberVec
    
    ### YOUR CODE ENDS HERE ###

    return aNumVec


### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
#     Notes:
#     The scale factor should be the number of leading zeros

    ### YOUR CODE STARTS HERE ###
    
    Rs = nScaleBits
    Rm = nMantBits
        
    # get number of R bits
    R = 2**(Rs) - 1 + Rm
    
    # get number of bits
    nBits = R
    
    # quantize aNum
    aQuantizedNum = QuantizeUniform(aNum, nBits)

    # create mask for getting the |code|
    mask = ~(1 << nBits - 1)
    # AND mask with aQuantizedNum to get |code|
    abs_code = aQuantizedNum & mask

    # create mask to count number of leading zeros
    and_mask = 1 << nBits - 1
    # discard the first bit (that used to be the sign bit) to count # of leading zeros in |code|
    abs_code_temp = abs_code << 1
    # count number of leading zeros 
    num_zeros = 0
    curr_bit = 0
    
    while((abs_code_temp & and_mask == 0) and (curr_bit < nBits - 1)):
        abs_code_temp = abs_code_temp << 1
        num_zeros += 1
        curr_bit += 1
    
    if num_zeros < 2**(Rs) - 1:
        scale = num_zeros
    else:
        scale = 2**(Rs)-1

    ### YOUR CODE ENDS HERE ###

    return scale



### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
       
    Rs = nScaleBits
    Rm = nMantBits
        
    # get number of bits
    nBits = 2**(Rs)-1 + Rm
    
    # quantize aNum
    aQuantizedNum = QuantizeUniform(aNum,nBits)
    # get s value
    s = aQuantizedNum >> (nBits - 1)
    # get |code| value
    mask = ~(1 << nBits - 1)
    abs_code = aQuantizedNum & mask

    # set mantissa bits
    if scale == 2**(Rs)-1:
        mantissa_s = s << (Rm - 1)
        mantissa_code = abs_code & ~(0 << (Rm-1-1))
        mantissa = mantissa_s | mantissa_code
        
    else:
        mantissa_s = s << (Rm - 1)
        
        # create mask to count number of leading zeros
        and_mask = 1 << nBits - 1
        # discard the first bit (that used to be the sign bit) to count # of leading zeros in |code|
        abs_code_temp = abs_code << 1
        num_zeros = 0
        curr_bit = 0
        while((abs_code_temp & and_mask == 0) and (curr_bit < nBits - 1)):
            abs_code_temp = abs_code_temp << 1
            num_zeros += 1
            curr_bit += 1
        
        # compute mantissa_code 
        bits_after_leading_one = nBits - num_zeros - 2 # omitting the one following leading zeros
        code_mask =  ~(1 << bits_after_leading_one)
        
        shift = bits_after_leading_one - (Rm - 1)
        mantissa_code = (code_mask & abs_code) >> shift

        # compute mantissa
        mantissa = mantissa_s | mantissa_code


    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """

    aNum = 0.0 # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    ### YOUR CODE STARTS HERE ###
    Rs = nScaleBits
    Rm = nMantBits
    
    # define R
    R = 2**(Rs)-1 + Rm
    
    # compute number of leading zeros
    num_zeros = scale
    
    # get sign
    sign = mantissa & (1 << Rm-1)
    sign_num = sign << (R-Rm)
    
    # get last Rm-1 mantissa bits
    mask = ~(1 << (Rm-1))
    mantissa_code = mantissa & mask

    
    # compute aQuantizedNum
    if scale == 2**(Rs)-1:
        abs_code = mantissa_code
        aQuantizedNum = sign_num | mantissa_code

    else:
        mask = 1 << (Rm-1) 
        abs_code = mantissa_code | mask # add leading one before mantissa code
        if num_zeros < 2**(Rs)-2:
            abs_code = (abs_code << 1) | 1 # add trailing one
            abs_code = abs_code << (R - Rm - 2 - num_zeros)# add trailing zeros
            
        aQuantizedNum = sign_num | abs_code
    
    # dequantize the aQuantizedNum
    aNum = DequantizeUniform(aQuantizedNum,R)
        

    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    
    Rs = nScaleBits
    Rm = nMantBits
        
    # get number of bits
    nBits = 2**(Rs)-1 + Rm
    
    # quantize aNum
    aQuantizedNum = QuantizeUniform(aNum,nBits)
    # get s value
    s = aQuantizedNum >> (nBits - 1)
    # get |code| value
    mask = ~(1 << nBits - 1)
    abs_code = aQuantizedNum & mask

    # set mantissa bits
    if scale == 2**(Rs)-1:
        mantissa_s = s << (Rm - 1) 
        mantissa_code = abs_code & ~(0 << (Rm-1-1))
        mantissa = mantissa_s | mantissa_code
        
    else:
        mantissa_s = s << (Rm - 1)
        
        # create mask to count number of leading zeros
        and_mask = 1 << nBits - 1
        # discard the first bit (that used to be the sign bit) to count # of leading zeros in |code|
        abs_code_temp = abs_code << 1
        num_zeros = 0
        curr_bit = 0
        while((abs_code_temp & and_mask == 0) and (curr_bit < nBits - 1)):
            abs_code_temp = abs_code_temp << 1
            num_zeros += 1
            curr_bit += 1
        
        # compute mantissa_code 
        bits_after_leading_one = nBits - num_zeros - 1 # don't omit the one following leading zeros
        code_mask =  ~(1 << bits_after_leading_one)
        shift = bits_after_leading_one - (Rm - 1)
        mantissa_code = (code_mask & abs_code) >> shift

        # compute mantissa
        mantissa = mantissa_s | mantissa_code



    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    
    Rs = nScaleBits
    Rm = nMantBits
    
    # define R
    R = 2**(Rs)-1 + Rm
    
    # compute number of leading zeros
    num_zeros = scale
    
    # get sign
    sign = mantissa & (1 << Rm-1)
    sign_num = sign << (R-Rm)
    
    # get last Rm-1 mantissa bits
    mask = ~(1 << (Rm-1))
    mantissa_code = mantissa & mask
    
    # compute aQuantizedNum
    if scale == 2**(Rs)-1:
        abs_code = mantissa_code
        aQuantizedNum = sign_num | mantissa_code
    
    
    else:
        abs_code = mantissa_code

        abs_code = (abs_code << 1) | 1 # add trailing one
        abs_code = abs_code << (R - Rm - 1 - num_zeros)# add trailing zeros
        
        aQuantizedNum = sign_num | abs_code
    
    # dequantize the aQuantizedNum
    aNum = DequantizeUniform(aQuantizedNum,R)


    ### YOUR CODE ENDS HERE ###

    return aNum


### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    mantissaVec = np.zeros_like(aNumVec, dtype = int) # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    ### YOUR CODE STARTS HERE ###
    
    Rs = nScaleBits
    Rm = nMantBits

    # get number of bits
    nBits = 2**(Rs)-1 + Rm
    
    # quantize aNum
    aQuantizedNumVec = vQuantizeUniform(aNumVec,nBits)
    
    # get s value
    sVec = aQuantizedNumVec >> (nBits - 1)
    
    # get |code| value
    mask = ~(1 << nBits - 1)
    abs_codeVec = aQuantizedNumVec & mask
        
    # set mantissa bits
    mantissa_sVec = sVec << (Rm - 1)
    shift = nBits - 1 - scale - (Rm - 1)
    mantissa_codeVec = abs_codeVec >> shift

    # compute mantissa
    mantissaVec = mantissa_sVec | mantissa_codeVec


    ### YOUR CODE ENDS HERE ###

    return mantissaVec


### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    Rs = nScaleBits
    Rm = nMantBits
    
    # define R
    R = 2**(Rs)-1 + Rm
    
    # compute number of leading zeros
    num_zeros = scale
    
    # get sign
    signVec = mantissaVec & (1 << Rm-1)
    sign_numVec = signVec << (R-Rm)
    
    # get last Rm-1 mantissa bits
    mask = ~(1 << (Rm-1))
    mantissa_codeVec = mantissaVec & mask
    
    # compute aQuantizedNum  

    abs_codeVec = mantissa_codeVec
    abs_codeVec = abs_codeVec << (R - Rm - num_zeros)# add trailing zeros

    aQuantizedNumVec = sign_numVec | abs_codeVec
    
    # dequantize the aQuantizedNum
    aNumVec = vDequantizeUniform(aQuantizedNumVec,R)  
    
    ### YOUR CODE ENDS HERE ###

    return aNumVec

#  

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###

    pass # THIS DOES NOTHING

    ### YOUR TESTING CODE ENDS HERE ###