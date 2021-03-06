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
    
    bits = np.zeros((nBands), dtype = int)
    if len(SMR) ==0:
        return bits

    # first pass allocation
    while(bitBudget >= np.min(nLines)):  # while there are enough bits to allocate   
        ind_max = np.argmax(SMR) #index of largest SMR value
        if(bitBudget >= nLines[ind_max]): # if have enough bits for current SMR band
            bits[ind_max] += 1 # allocate a bit in the given indx
            bitBudget -= nLines[ind_max] # -1 in our bit budget
        SMR[ind_max] -= 6.02 # adjust SMR value
    
    # if bits per band > maxMantBits, make bits = maxMantBits and add to extras
    found_max = np.where(bits > maxMantBits) # boolean array
    extra_bits = bits[found_max] - maxMantBits
    bits[found_max] -= extra_bits
    bitBudget += np.sum(nLines[found_max]*extra_bits)

 
    #get rid of all ones in bits and reallocate 
    while (bits == 1).any() and (bits < maxMantBits).any():
        
        # get rid of all the ones
        single = np.max(np.argwhere(bits == 1))
        bits[single] -= 1
        bitBudget += np.sum(nLines[single])
        
        # reallocate 
        ind_max = np.argmax(SMR) #index of largest SMR value
        if (bitBudget >= nLines[ind_max] and bits[ind_max] > 0 and bits[ind_max] < maxMantBits):
            bits[ind_max] += 1 # allocate a bit in the given indx
            bitBudget -= nLines[ind_max] # -1 in our bit budget
        SMR[ind_max] -= 6.02 # adjust SMR value
        
    return bits  
    

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE






