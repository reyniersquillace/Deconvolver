import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from optuna.trial import Trial as trial
import logging as log

log.basicConfig(level=log.NOTSET)

class Model(object):

    def __init__(
                self,
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
                ):

        self.samples = samples
        self.features = features
        self.input_size = np.shape(samples)[1]
        self.output_size = 2 #one error for each input feature

        self.max_layers = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.device = Model.device()
        self.epochs = epochs
        self.batch_size = batch_size
        self.workers = workers
        self.g = g
        self.h = h
        self.min_valid = min_valid
        self.architecture = architecture

    def device():

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        log.info(f"Using {device} as device.")

        return device

    def MLP_architecture(self, trial, n_layers = 1):
        
        if n_layers == 1:

            n_neurons = 2048

        elif n_layers == 2:
            
            n_neurons = 1024

        mlp = nn.MLPClassifier(n_neurons)

        return mlp


   def __call__(self, trial):

        #define optimization parameters
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("wd", 1e-8, 1e0,  log=True)

        # define the optimizer
        optimizer = torch.optim.AdamW(
                                    model.parameters(),
                                    lr = learning_rate,
                                    betas=(0.5, 0.999),
                                    weight_decay = weight_decay
                                    )
        loss = nn.CrossEntropyLoss()

        trial_file   = './class_losses/loss_%d.txt'%(trial.number)
        model_file = './class_models/model_%d.pt'%(trial.number)

        #split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.samples, self.features, train_size=0.7, shuffle=True)

        #cast data as torch tensor with correct shape and data type
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        for epoch in range(self.epochs):
            #????
            train_loss1, train_loss = torch.zeros(len(self.g)).to(self.device), 0.0
            train_loss2, points     = torch.zeros(len(self.g)).to(self.device), 0

            model.train()

            for X_train, y_train in zip(X_train, y_train):
                bs = X_train.shape[0]
                X = X_train.to(self.device)
                y = y_train.to(self.device)
                p = model(X)

                #get posterior mean and error
                #y_NN = p[:,self.g].squeeze()
                #e_NN = p[:,self.h].squeeze()
                y_NN = p[:].squeeze()
                e_NN = p[:].squeeze()

                #get loss of mean and error
                loss1 = loss(y_NN, y)
                loss2 = loss(e_NN, loss1)

                #get total loss
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))

                #get training run losses overall
                train_loss1 += loss1*bs
                train_loss2 += loss2*bs

                points += bs

                #fancy machine learning stuff
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
            train_loss = torch.mean(train_loss).item()

            #now we move on to validation
            valid_loss1, valid_loss = torch.zeros(len(self.g)).to(self.device), 0.0
            valid_loss2, points     = torch.zeros(len(self.g)).to(self.device), 0

            model.eval()

            for X_test, y_test in zip(X_test, y_test):
                with torch.no_grad():

                    bs = X_test.shape[0]
                    X = X_test.to(self.device)
                    #X = torch.Tensor([[X]])
                    y = y_test.to(self.device)
                    p = model(X)

                    #get posterior mean and error
                    #y_NN = p[:,self.g].squeeze()
                    #e_NN = p[:,self.h].squeeze()

                    y_NN = p[:].squeeze()
                    e_NN = p[:].squeeze()

                    #get loss of mean and error
                    loss1 = loss(y_NN, y)
                    loss2 = loss(e_NN, loss1)
                    #get total loss
                    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))

                    #get training run losses overall
                   # valid_loss1 += loss1*self.batch_size
                   # valid_loss2 += loss2*self.batch_size
                    valid_loss1 += loss1*bs
                    valid_loss2 += loss2*bs
                   # points     += self.batch_size
                    points += bs
                valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
                valid_loss = torch.mean(valid_loss).item()
              # save best model if found
                if valid_loss<self.min_valid:

                    self.min_valid = valid_loss

                    torch.save(model, model_file)
                    #torch.save(model.state_dict(), model_file)

                print('')



                f = open(trial_file, 'a')
                f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
                f.close()


            return self.min_valid

