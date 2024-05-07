import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time

# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, f_Pk, f_Pk_norm, f_params, splits):

        # read data, scale it, and normalize it
        Pk = np.log10(np.load(f_Pk))
        if f_Pk_norm is None:
            mean, std = np.mean(Pk, axis=0), np.std(Pk, axis=0)
        else:
            Pk_norm = np.log10(np.load(f_Pk_norm))
            mean, std = np.mean(Pk_norm, axis=0), np.std(Pk_norm, axis=0)
        Pk = (Pk - mean)/std

        # read the value of the cosmological & astrophysical parameters; normalize them
        params  = np.loadtxt(f_params)
        #minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        #maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])

        params_maps = np.zeros((params.shape[0]*splits), dtype=np.float32)
        for i in range(params.shape[0]):
            for j in range(splits):
                params_maps[i*splits + j] = params[i]

        minimum = np.array([0.03333])
        maximum = np.array([0.4])
        params  = (params - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        sims = params.shape[0]
        if   mode=='train':  size, offset = int(sims*0.90), int(sims*0.00)
        elif mode=='valid':  size, offset = int(sims*0.05), int(sims*0.90)
        elif mode=='test':   size, offset = int(sims*0.05), int(sims*0.95)
        elif mode=='all':    size, offset = int(sims*1.00), int(sims*0.00)
        else:                raise Exception('Wrong name!')
        
        # randomly shuffle the sims. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        sim_numbers = np.arange(sims)

        np.random.shuffle(sim_numbers)
        sim_numbers = sim_numbers[offset:offset+size] #select indexes of mode

        indexes = np.zeros(sims*splits, dtype=np.int32)
        count = 0
        for i in sim_numbers:
            for j in range(splits):
                indexes[count] = i*splits + j
                count += 1

        # select the data in the considered mode
        Pk     = Pk[indexes]
        params = params_maps[indexes]

        # define size, input and output matrices
        self.size   = size * splits
        self.input  = torch.tensor(Pk,     dtype=torch.float)
        self.output = torch.tensor(params, dtype=torch.float)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
# mode ---------------> 'train', 'valid', 'test' or 'all'
# seed ---------------> random seed to split data among training, validation and testing
# f_Pk ---------------> file containing the power spectra
# f_Pk_norm ----------> file containing the power spectra to normalize data
# f_params -----------> files with the value of the cosmological + astrophysical params
# batch_size ---------> batch size
# shuffle ------------> whether to shuffle the data or not
# workers --------> number of CPUs to load the data in parallel
def create_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, batch_size, shuffle, workers, splits):
    data_set = make_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, splits)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers)
