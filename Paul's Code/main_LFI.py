import numpy as np

import sys, os, time

import torch

import torch.nn as nn

import data, architecture

import optuna





class Objective(object):

    def __init__(self, input_size, output_size, max_layers, max_neurons_layers, device,

                 epochs, seed, batch_size, workers, splits):



        self.input_size         = input_size

        self.output_size        = output_size

        self.max_layers         = max_layers

        self.max_neurons_layers = max_neurons_layers

        self.device             = device

        self.epochs             = epochs

        self.seed               = seed

        self.batch_size         = batch_size

        self.workers            = workers



    def __call__(self, trial):



        # name of the files that will contain the losses and model weights

        fout   = 'losses/loss_%d.txt'%(trial.number)

        fmodel = 'models/model_%d.pt'%(trial.number)



        # generate the architecture

        model = architecture.dynamic_model(trial, self.input_size, self.output_size, 

                            self.max_layers, self.max_neurons_layers).to(self.device)



        # get the weight decay and learning rate values

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        wd = trial.suggest_float("wd", 1e-8, 1e0,  log=True)



        # define the optimizer

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 

                                      weight_decay=wd)



        # get the data

        train_loader = data.create_dataset('train', self.seed, f_Pk, f_Pk_norm, 

                                           f_params, self.batch_size, 

                                           shuffle=True, workers=self.workers, splits=splits)

        valid_loader = data.create_dataset('valid', self.seed, f_Pk, f_Pk_norm, 

                                           f_params, self.batch_size, 

                                           shuffle=False, workers=self.workers, splits=splits)



        # train/validate model

        min_valid = 1e40

        for epoch in range(self.epochs):



            # do training

            train_loss1, train_loss = torch.zeros(len(g)).to(device), 0.0

            train_loss2, points     = torch.zeros(len(g)).to(device), 0

            model.train()

            for x, y in train_loader:

                bs   = x.shape[0]         #batch size

                x    = x.to(device)       #maps

                #y    = y.to(device)[:,g]  #parameters

                y    = y.to(device)  #parameters

                p    = model(x)           #NN output

                #y_NN = p[:,g]             #posterior mean

                #e_NN = p[:,h]

                y_NN = p[:,g].squeeze()             #posterior mean

                e_NN = p[:,h].squeeze()             #posterior std

                #print(y.cpu().shape, y_NN.cpu().shape, e_NN.cpu().shape)

                loss1 = torch.mean((y_NN - y)**2,                axis=0)

                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)

                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))

                train_loss1 += loss1*bs

                train_loss2 += loss2*bs

                points      += bs

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)

            train_loss = torch.mean(train_loss).item()



            # do validation

            valid_loss1, valid_loss = torch.zeros(len(g)).to(device), 0.0

            valid_loss2, points     = torch.zeros(len(g)).to(device), 0

            model.eval()

            for x, y in valid_loader:

                with torch.no_grad():

                    bs    = x.shape[0]         #batch size

                    x     = x.to(device)       #maps

                    #y     = y.to(device)[:,g]  #parameters

                    y     = y.to(device)  #parameters

                    p     = model(x)           #NN output

                    #y_NN  = p[:,g]             #posterior mean

                    #e_NN  = p[:,h]

                    y_NN  = p[:,g].squeeze()             #posterior mean

                    e_NN  = p[:,h].squeeze()             #posterior std

                    loss1 = torch.mean((y_NN - y)**2,                axis=0)

                    loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)

                    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))

                    valid_loss1 += loss1*bs

                    valid_loss2 += loss2*bs

                    points     += bs

            valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)

            valid_loss = torch.mean(valid_loss).item()



            print('%03d %.3e %.3e '%(epoch, train_loss, valid_loss), end='')



            # save best model if found

            if valid_loss<min_valid:  

                min_valid = valid_loss

                torch.save(model.state_dict(), fmodel)

                print('(C) ', end='')

            print('')



            f = open(fout, 'a')

            f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))

            f.close()



            # Handle pruning based on the intermediate value

            # comment out these lines if using prunning

            #trial.report(min_valid, epoch)

            #if trial.should_prune():  raise optuna.exceptions.TrialPruned()



        return min_valid



##################################### INPUT ##########################################



res = 256

sim_res = 256

redshift = 0



# data parameters

f_Pk      = f'output/power_spec/Pk_res{res}_sim{sim_res}_z{redshift}_{res//256*32}k.npy'

f_Pk_norm = None

f_params  = 'output/mass.txt' 

seed      = 1

splits    = 15



# architecture parameters

input_size         = 126 #number of bins in Pk

output_size        = 2  #number of parameters to predict (posterior mean + std)

max_layers         = 5

max_neurons_layers = 1000



# training parameters

batch_size = 32

epochs     = 200

workers    = 2     #number of cpus to load the data 

g          = [0]  #minimize loss using parameters 0 and 1

h          = [1]  #minimize loss using errors of parameters 0 and 1



# optuna parameters

study_name       = f'Pk_2_params_res{res}_sim{sim_res}_z{redshift}'

n_trials         = 50 #set to None for infinite

storage          = 'sqlite:///output/databases/Pk_dmo_better.db'

n_jobs           = 1

n_startup_trials = 20 #random sample the space before using the sampler

######################################################################################



# use GPUs if available

if torch.cuda.is_available():

    print("CUDA Available")

    device = torch.device('cuda')

else:

    print('CUDA Not Available')

    device = torch.device('cpu')



# create output folders if they dont exist

for fout in ['models', 'losses']:

    if not(os.path.exists(fout)):  os.system('mkdir %s'%fout)



# define the optuna study and optimize it

objective = Objective(input_size, output_size, max_layers, max_neurons_layers, 

                      device, epochs, seed, batch_size, workers, splits)

sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)

study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage,

                            load_if_exists=True)

study.optimize(objective, n_trials, n_jobs=n_jobs)
