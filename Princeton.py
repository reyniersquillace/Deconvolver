import numpy as np

#this function is modified from Scott Ransom's PRESTO suite

def fft_rotate(arr, bins):
    """
    fft_rotate(arr, bins):
        Return array 'arr' rotated by 'bins' places to the left.  The
            rotation is done in the Fourier domain using the Shift Theorem.
            Unlike the accurate Princeton convention, 'bins' cannot be fractional.  
            The resulting vector will have the same length as the original.
    """
    arr = np.asarray(arr)
    freqs = np.arange(arr.size / 2 + 1, dtype=float)
    phasor = np.exp(complex(0.0, 2*np.pi) * freqs * bins / float(arr.size))
    return np.fft.irfft(phasor * np.fft.rfft(arr), arr.size)

def pseudo_princeton(pulse, locs, center = 0.5):
    n = len(pulse)
    ftpulse = np.fft.rfft(pulse)
    phi0 = np.angle(ftpulse[1])
    newpulse = fft_rotate(pulse, -int((phi0)/(2*np.pi)*n+center*n))
    newlocs = np.roll(locs, int((phi0)/(2*np.pi)*n + center*n))
    return newpulse, newlocs
