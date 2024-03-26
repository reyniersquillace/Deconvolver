import numpy as np
import numpy.random as rand

def lorentzian(phi, gamma, x0 = 0.5):
    return (gamma/((phi - x0)**2 + gamma**2))/np.pi


def generate(n_samples, n_bins, noise = True):
    max_subpulses = 5
    phi = np.linspace(0, 1, n_bins)
    pulses = np.zeros((n_samples, n_bins))
    locs = np.zeros((n_samples, max_subpulses + 1))
    gammas = np.copy(locs)
    amps = np.copy(locs)

    for i in range(n_samples):
        gamma = rand.choice(np.linspace(0.001, 0.1, 1000))
        pulse = lorentzian(phi, gamma)
        pulse /= max(pulse)
        

        subpulses = rand.randint(0, max_subpulses + 1)

        #subpulses
        for j in range(subpulses):
            gamma_sub = rand.choice(np.linspace(0.001, 0.1, 1000))
            loc_sub = 0.5 + rand.normal(0, 0.1)
            subpulse = lorentzian(phi, gamma_sub,loc_sub)
            amp = rand.choice(np.linspace(0.1, 0.9))
            subpulse *= amp/max(subpulse)
            pulse += subpulse

            locs[i, j + 1] = loc_sub
            gammas[i, j + 1] = gamma_sub
            amps[i, j + 1] = amp

        if noise:
            rms = rand.choice(np.linspace(0.001, 0.3, 3000))
            pulse += rand.normal(0, rms, n_bins)

        #pulse /= max(pulse) 
        pulses[i] = pulse
        locs[i, 0] = 0.5
        gammas[i, 0] = gamma
        amps[i, 0] = 1

    return pulses, locs, gammas, amps