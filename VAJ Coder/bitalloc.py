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
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR, bitReservoir):
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
    
    bits = np.zeros(nBands, dtype=int)


    if len(SMR) ==0:
        return bits
    
    allocateBits = True
    bits_used = 0
    threshold = -12
    maxBitBudget = 8000

    # first pass allocation
    
    # allocate bits of lower bands  
    while(allocateBits):
        if (bitBudget >= np.min(nLines[0:12])): # while there are enough bits to allocate 
            ind_max = np.argmax(SMR[0:12])  #index of largest SMR value
            if(bitBudget >= nLines[ind_max]): # if have enough bits for current SMR band
                if(SMR[ind_max] >= threshold): # only allocate bits if SMR is greater than threshold
                    bits[ind_max] += 1 # allocate a bit in the given indx
                    bitBudget -= nLines[ind_max] # -1 in our bit budget
                    bits_used += 1*nLines[ind_max]
            SMR[ind_max] -= 6.02 # adjust SMR value
            
        if max(SMR[0:12]) < threshold or bitBudget < np.min(nLines[0:12]): # break loop if all SMR values are negative or if run out of bits
            allocateBits = False
    
    allocateBits = True
    # allocate bits of upper bands  
    while(allocateBits):
        if (bitBudget >= np.min(nLines[12:])): # while there are enough bits to allocate 
            ind_max = np.argmax(SMR[12:]) + 12  #index of largest SMR value
            if(bitBudget >= nLines[ind_max]): # if have enough bits for current SMR bands
                if(SMR[ind_max] >= threshold): # only allocate bits if SMR is greater than threshold
                    bits[ind_max] += 1 # allocate a bit in the given indx
                    bitBudget -= nLines[ind_max] # -1 in our bit budget
                    bits_used += 1*nLines[ind_max]
            SMR[ind_max] -= 6.02 # adjust SMR value
            
        if max(SMR[12:]) < threshold or bitBudget < np.min(nLines[12:]): # break loop if all SMR values are negative or if run out of bits
            allocateBits = False
   
        
    # get rid of all ones in bits and reallocate
    single = np.where(bits == 1) # boolean array
    bits[single] -= 1
    bitBudget += np.sum(nLines[single])
    
    # if bits per band > maxMantBits, make bits = maxMantBits and add to extras
    too_many = np.where(bits > maxMantBits) # boolean array
    extra_bits = bits[too_many] - maxMantBits
    bits[too_many] -= extra_bits
    bitBudget += np.sum(nLines[too_many]*extra_bits)
    
    # reallocate bits
    allocateBits = True
    while(allocateBits):
        if(bitBudget >= np.min(nLines)):
            ind_max = np.argmax(SMR)
            if(bitBudget >= nLines[ind_max] and bits[ind_max] > 0 and bits[ind_max] < maxMantBits):
                if(SMR[ind_max] >= threshold): 
                    bits[ind_max] += 1
                    bitBudget -= nLines[ind_max]
                    bits_used += 1*nLines[ind_max]
                if(bitBudget >= maxBitBudget):
                    bits[ind_max] += 1
                    bitBudget -= nLines[ind_max]
                    bits_used += 1*nLines[ind_max]
                
            SMR[ind_max] -= 6.02
            
        if max(SMR) < threshold or bitBudget < np.min(nLines): # break loop if all SMR values are negative or if run out bits
            allocateBits = False     
   
    
    bitReservoir = bitBudget;
  
     
#     if bits_used > 1281:
#         print("bits used", bits_used)
#         print("bit budget - after", bitBudget); 
    
      
    return bits, bitReservoir


   

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE







