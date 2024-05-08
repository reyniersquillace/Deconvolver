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
                architecture,
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


    def LNN_architecture(self, trial):
        '''
        This function creates the model architecture for a linear NN using optuna.

            Inputs:
            -------
                    trial (optuna Trial object): the trial object for the current run

            Returns:
            --------
                    pytorch Sequential object
        '''
        layers = []
        
        #use optuna to predict best number of hidden layers
        n_layers = trial.suggest_int("n_layers", 1, self.max_layers)
        log.info(f"Created a model with {n_layers} hidden layers.")
        
        in_features = self.input_size

        for i in range(n_layers):
             
            #use optuna to predict best output size
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, self.max_neurons_layers)
            
            #create layer and add it to layers
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU(0.2))
            
            #use optuna to predict best train/validate split
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.8)
            
            log.info(f"Created a hidden layer of size ({in_features}, {out_features}).")
            #add layer to layers
            layers.append(nn.Dropout(p))
            
            #set new input size to current output size
            in_features = out_features
    
        # get the last layer
        layers.append(nn.Linear(out_features, self.output_size))
        #create and return model
        return nn.Sequential(*layers)
    
        
    def RNN_architecture(self, trial):
        
        #use optuna to predict best number of hidden layers
        n_layers = trial.suggest_int("n_layers", 1, self.max_layers)
        log.info(f"Created a model with {n_layers} hidden layers.")
        
        in_features = self.input_size
        out_features = trial.suggest_int("n_units_l{}".format(1), 4, self.max_neurons_layers)
        
        rnn = nn.GRU(in_features)
    
        return rnn

    def __call__(self, trial):
        #create model
        if self.architecture == 'rnn' or self.architecture == 'RNN':
            model = Model.RNN_architecture(self, trial).to(self.device)
        elif self.architecture == 'lnn' or self.architecture == 'LNN':
            model = Model.LNN_architecture(self, trial).to(self.device)
        else:
            raise(f"{architecture} is not a valid architecture.")

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
       
        trial_file   = './{architecture}_losses/loss_%d.txt'%(trial.number)
        model_file = './{architecture}_conv_models/model_%d.pt'%(trial.number)

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
                loss1 = torch.mean((y_NN - y)**2, axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)

                #get total loss
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))

                #get training run losses overall
                train_loss1 += loss1*self.batch_size
                train_loss2 += loss2*self.batch_size

                points += self.batch_size

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
                    #y = torch.Tensor([[y]])
                    p = model(X)
                    
                    #get posterior mean and error
                    #y_NN = p[:,self.g].squeeze()
                    #e_NN = p[:,self.h].squeeze()

                    y_NN = p[:].squeeze()
                    e_NN = p[:].squeeze()
                    
                    #get loss of mean and error
                    loss1 = torch.mean((y_NN - y)**2, axis=0)
                    loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)

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



                print('%03d %.3e %.3e '%(epoch, train_loss, valid_loss), end='')



                # save best model if found
                if valid_loss<self.min_valid:  

                    self.min_valid = valid_loss
                    
                    torch.save(model, model_file)
                    #torch.save(model.state_dict(), model_file)
                    print('(C) ', end='')

                print('')



                f = open(trial_file, 'a')
                f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
                f.close()


            return self.min_valid
