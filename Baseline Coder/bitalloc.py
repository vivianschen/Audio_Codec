import numpy as np

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """

    return np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    return np.array([9, 10, 11, 15, 17, 15, 15, 15, 11,  8,  7,  7,  6,  5,  5,  5,  4, 10, 13,  3,  3, 13,  3,  0, 0])

def BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    return np.array([7, 12, 12, 12, 10,  9,  9, 10,  7,  7,  9, 10, 11, 10, 10, 10,  6,  8, 11,  3,  3, 10,  2,  1, 0])

# Question 1.c)
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Returns:
            bits[nBands] is number of bits allocated to each scale factor band
            
         Logic:
           Maximizing SMR over blook gives optimization result that:
               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
           where P is the pool of bits for mantissas and N is number of bands
           This result needs to be adjusted if any R(i) goes below 2 (in which
           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
           We will not bother to worry about slight variations in bit budget due
           rounding of the above equation to integer values of R(i).   
    """
    DBTOBITS = 6.02
    
    # initial bit allocation using water-filling algorithm
    bits = np.zeros(nBands , dtype=int)
    valid = np.ones(nBands , dtype=bool)
    while valid.any():
        i = np.arange(nBands)[valid][np.argmax((SMR - bits*DBTOBITS)[valid])]
        if (bitBudget - nLines[i]) >= 0:
            bits[i] += 1
            bitBudget -= nLines[i]
            if bits[i] >= maxMantBits:
                valid[i] = False 
        else:
            valid[i] = False
    
    # reassign lonely bits
    valid = bits < maxMantBits
    while (bits == 1).any() and valid.any():
        # lonely bit bands and pick the highest if there are more than 1
        i = np.max(np.argwhere(bits==1)) 
        bits[i] = 0
        valid[i] = False
        
        i = np.arange(nBands)[valid][np.argmax((SMR - bits*DBTOBITS)[valid])] 
        if (bitBudget - nLines[i]) >= 0:
            bits[i] += 1
            bitBudget -= nLines[i]
            if bits[i] >= maxMantBits:
                valid[i] = False
        else:
            valid[i] = False
                
    return bits
    
#     # take average SMR
#     avgSMR = np.mean(SMR)
#     # initiliaze bit allocation array
#     bits = np.zeros((nBands), dtype = int)
#     # allocate bits 
#     for i in range(nBands):
#         bits[i] = int((bitBudget / nLines[i]) + 1/6.02 * (SMR[i] - avgSMR)) 
    
#     # if bits is great tha max bits set it to max bits
#     bits = np.where(bits > maxMantBits, maxMantBits, bits)
# #     bits[bits > maxMantBits] = maxMantBits
#     # if bits is 1 or less set bits to zero
#     bits = np.where(bits <= 1, 0, bits)
# #     bits[bits <= 1] = 0
    
#     bits = bits.tolist()
    
# #     print("Bits: ", bits)

    
    return bits # TO REPLACE WITH YOUR CODE

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE






