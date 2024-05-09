import numpy as np
import numpy.random as rand

def lorentzian(phi, gamma, x0 = 512):
    return (gamma/((phi - x0)**2 + gamma**2))/np.pi


def generate(n_samples, n_bins, noise = False):
    max_subpulses = 4
    phi = np.arange(n_bins, dtype = int)
    pulses = np.zeros((n_samples, n_bins))
    locs = np.zeros((n_samples, n_bins)) 
    gammas = np.copy(locs)

    for i in range(n_samples):
        gamma = rand.choice(np.linspace(1, 50, n_bins))
        pulse = lorentzian(phi, gamma)
        pulse /= max(pulse)

        subpulses = rand.randint(0, max_subpulses + 1)
        subpulses = max_subpulses 
        #subpulses
        for j in range(subpulses):
            gamma_sub = rand.choice(np.linspace(1, 50, n_bins))
            loc_sub = 512 + rand.randint(-200, 200)
            subpulse = lorentzian(phi, gamma_sub,loc_sub)
            amp = rand.choice(np.linspace(0.1, 0.9))
            subpulse *= amp/max(subpulse)
            pulse += subpulse

            locs[i, loc_sub] = 1
            pulses[i] = pulse
            gammas[i, j + 1] = gamma_sub

        if noise:
            rms = rand.choice(np.linspace(0.001, 0.1, 3000))
            pulse += rand.normal(0, rms, n_bins)

        pulses[i] = pulse
        locs[i, 512] = 1
        gammas[i, 0] = gamma

    return pulses, locs, gammas

def generate_dummy(n_samples, n_bins):
    phi = np.linspace(0, 1, 1024)
    pulses = np.zeros((n_samples, n_bins))
    locs = np.zeros(n_samples)

    for i in range(n_samples):

        pulse = np.zeros(1024)

        gamma_sub = rand.choice(np.linspace(0.001, 0.05, 1000))
        loc_sub = rand.choice(np.linspace(0.1, 0.9, 1024))
        subpulse = lorentzian(phi, gamma_sub,loc_sub)
        amp = rand.choice(np.linspace(0.1, 0.9))
        subpulse *= amp/max(subpulse)
        pulse += subpulse

        locs[i] = loc_sub
        pulses[i] = pulse

    return pulses, locs

def generate_class(n_samples, n_bins):
    phi = np.arange(n_bins, dtype = int)
    pulses = np.zeros((n_samples, n_bins))
    locs = np.zeros((n_samples, n_bins))
    
    for i in range(n_samples):

        pulse = np.zeros(n_bins)

        gamma_sub = rand.choice(np.linspace(1, 50, 1024))
        loc_sub = rand.randint(0, 1024)
        subpulse = lorentzian(phi, gamma_sub,loc_sub)
        amp = rand.choice(np.linspace(0.1, 0.9))
        subpulse *= amp/max(subpulse)
        pulse += subpulse

        locs[i, loc_sub] = 1
        pulses[i] = pulse

    return pulses, locs


pulses, locs, gammas = generate(50000, 1024)
np.savez('Data.npz', *[pulses, locs, gammas], **{'pulses' : pulses, 'locs' : locs, 'gammas' : gammas})
