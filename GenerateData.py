import numpy as np
import numpy.random as rand
import Princeton

def lorentzian(phi, gamma, x0 = 512):
    return (gamma/((phi - x0)**2 + gamma**2))/np.pi


def generate(n_samples, n_bins, noise = True):
    max_subpulses = 6
    phi = np.arange(n_bins, dtype = int)
    pulses = np.zeros((n_samples, n_bins))
    locs = np.zeros((n_samples, n_bins)) 
    gammas = np.copy(locs)

    for i in range(n_samples):
        gamma = rand.choice(np.linspace(1, 50, 1024))
        pulse = lorentzian(phi, gamma)
        pulse /= max(pulse)
        pulse = np.zeros(1024)

        subpulses = rand.randint(0, max_subpulses + 1)
        subpulses = max_subpulses 
        #subpulses
        for j in range(subpulses):
            gamma_sub = rand.choice(np.linspace(1, 50, 1024))
            loc_sub = 512 + rand.randint(-200, 200)
            subpulse = lorentzian(phi, gamma_sub,loc_sub)
            amp = rand.choice(np.linspace(0.1, 0.9))
            subpulse *= amp/max(subpulse)
            pulse += subpulse

            locs[i, j + 1] = loc_sub
            gammas[i, j + 1] = gamma_sub

        if noise:
            rms = rand.choice(np.linspace(0.001, 0.3, 3000))
            pulse += rand.normal(0, rms, n_bins)

        pulses[i] = pulse
        locs[i, 0] = 512
        gammas[i, 0] = gamma
        amps[i, 0] = 1

    return pulses, locs, gammas, amps

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


pulses, locs = generate_class(20000, 1024)
np.savez('ClassData.npz', *[pulses, locs], **{'pulses' : pulses, 'locs' : locs})
