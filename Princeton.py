import numpy as np

#this function is modified from Scott Ransom's PRESTO suite

def fft_rotate(arr, bins):
    """
    Rotates an array to the left by a given number of bins. Unlike the accurate
    Princeton Convention, [bins] cannot be fractional.

        arr (arr) : the array to be rotated
        bins (int): the number of places by which to rotate it
    """
    arr = np.asarray(arr)
    freqs = np.arange(arr.size / 2 + 1, dtype=float)
    phasor = np.exp(complex(0.0, 2*np.pi) * freqs * bins / float(arr.size))
    return np.fft.irfft(phasor * np.fft.rfft(arr), arr.size)

def pseudo_princeton(pulse, locs, center = 0.5):
    '''
    This function performs the Princeton Convention as outlined in https://doi.org/10.1098/rsta.1992.0088.

    Inputs:
    ------
        pulse (arr)   : the pulse profile
        locs (arr)    : the locations of peaks in the training dataset
        center (float): where to align the weighted center of the pulse profile
    '''
    n = len(pulse)
    ftpulse = np.fft.rfft(pulse)
    phi0 = np.angle(ftpulse[1])
    newpulse = fft_rotate(pulse, -int((phi0)/(2*np.pi)*n+center*n))
    newlocs = np.roll(locs, int((phi0)/(2*np.pi)*n + center*n))
    return newpulse, newlocs
