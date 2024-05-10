import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import optuna
import logging as log
from ClassModel import Model

log.basicConfig(level=log.NOTSET)

#architecture parameters
max_layers         = 5
max_neurons_layers = 2000

#training parameters
output_size = 2 #posterior mean and error
batch_size = 32
epochs     = 200
workers    = 2     #number of cpus to load the data 
g          = [0]  #minimize loss using parameters 0 and 1
h          = [1]  #minimize loss using errors of parameters 0 and 1
min_valid = 1e40
# optuna parameters
study_name       = 'Shane_Project'
n_trials         = 20 #set to None for infinite
#storage          = './'
n_jobs           = 1
n_startup_trials = 20 #random sample the space before using the sampler

def args():
    '''
    This function parses the command line input.

    Returns:
    --------
        training_data (str.npz): the unsplit training set
        architecture (str)     : the model architecture to choose
    '''
    parser = argparse.ArgumentParser(
            prog = "PeakLNN",
            description = "Architecture options: LNN_preset, LNN_dynamic, RNN.",
            epilog = "Shane u better like my fancy user interface",
            )
    parser.add_argument('training_data')
    parser.add_argument('architecture')
    args = parser.parse_args()
    return args.training_data, args.outfile, args.architecture

def main():

    #load training data
    training_data, outfile, architecture = args()
    data = np.load(training_data)
    samples = data['pulses']
    features = data['locs']
    
    objective = Model(
                    samples,
                    features,
                    max_layers,
                    max_neurons_layers,
                    epochs,
                    batch_size,
                    workers,
                    g,
                    h,
                    min_valid,
                    architecture,
                    outfile,
                    )
    log.info(f"Creating sampler at t = {time.time() - start} s")
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    log.info(f"Creating study at t = {time.time() - start} s")
    study = optuna.create_study(study_name=study_name, sampler=sampler,
            load_if_exists=True)
    log.info(f"Optimizing study at t = {time.time() - start} s")
    study.optimize(objective, n_trials, n_jobs=n_jobs)
    log.info(f"Training completed at t = {time.time() - start} s")

tart = time.time()
log.info(f'Training and optimization started at {time.ctime(start)}')

main()

finish = time.time()
print(f'Training and optimization completed at {time.ctime(finish)}')

        
