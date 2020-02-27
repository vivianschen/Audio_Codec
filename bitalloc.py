import numpy as np

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """
    return np.array([4, 4, 4, 6, 4, 4, 6, 8, 6, 8, 10, 10, 12, 12, 18, 18, 24, 30, 38, 48, 54, 78, 106, 150, 362])

    
    #return 8*np.ones(nBands, dtype = np.int)# TO REPLACE WITH YOUR VECTOR

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    return np.array([16, 18, 20, 39, 32, 28, 42, 56, 30, 28, 30, 25, 30, 24, 36, 27, 36, 135, 228, 48, 27, 468, 106, 0, -362])


def BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    return np.array([16, 26, 26, 39, 22, 20, 30, 44, 24, 32, 50, 55, 72, 66, 99, 99, 84, 135, 228, 96, 108, 429, 159, 75, -905])


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

    # water-filling  algorithm
    bits = np.zeros(nBands, dtype=int)

    if len(SMR) ==0:
        return bits

    not_filled = np.ones(nBands, dtype=bool)

    # while there are still open bands
    while not_filled.any() and bitBudget > 0:
        # in nBands, get index of max SMR (i.e. my 5th band has greatest SMR)
        ind_max = np.arange(nBands)[not_filled][np.argmax((SMR - bits*6)[not_filled])]
        # get number of lines in this index. If bit budget minus this # lines >=0: 
        if (bitBudget - nLines[ind_max]) >= 0:
            bits[ind_max] += 1
            bitBudget -= nLines[ind_max]
            # if bits allocated to this index >= maximum, set index to filled 
            if bits[ind_max] >= maxMantBits:
                not_filled[ind_max] = False
        #if bit budget minus this # lines goes to negative, set index to filled
        else:
            not_filled[ind_max] = False

    # go back through and reallocate single bits or overflowing bits
    not_filled = bits < maxMantBits
    while (bits==1).any() and not_filled.any():
        # get the single bit in highest critical band, set back to 0
        i = np.max(np.argwhere(bits==1))
        bits[i] = 0
        bitBudget += nLines[i]

        # same thing...
        ind_max = np.arange(nBands)[not_filled][np.argmax(((SMR)-bits*6)[not_filled])]
        if (bitBudget - nLines[ind_max]) >= 0:
            bits[ind_max] += 1
            bitBudget -= nLines[ind_max]
        # set this index to filled
        not_filled[ind_max] = False

    #bits = np.minimum(bits, np.ones_like(bits)*maxMantBits)
    #print(bits)
    return bits

    

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE


# In[ ]:




