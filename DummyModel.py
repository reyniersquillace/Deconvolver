import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from optuna.trial import Trial as trial

class Model(self):

    def __init__(
                self, 
                samples, 
                features, 
                max_layers, 
                max_neurons_layers, 
                epochs, 
                batch_size, 
                workers,
                ):

        self.samples = samples
        self.features = features
        self.input_size = len(samples)
        self.output_size = len(features)*2 #one error for each input feature
        self.max_layers = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.device = Model.device()
        self.epochs = epochs
        self.batch_size = batch_size
        self.workers = workers
        self.splits = splits
    
    def device(self):

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    def make_file(self):

        for fout in ['models', 'losses']:
            if not(os.path.exists(fout)):
                os.system('mkdir %s'%fout)

    def architecture(self, trial):
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
    
        in_features = self.input_size

        for i in range(n_layers):
   
            #use optuna to predict best output size
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, self.max_neurons_layers)

            #create layer and add it to layers
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU(0.2))
            
            #use optuna to predict best train/validate split
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.8)

            #add layer to layers
            layers.append(nn.Dropout(p))
            
            #set new input size to current output size
            in_features = out_features
    
        # get the last layer
        layers.append(nn.Linear(out_features, output_size))

        #create and return model
        return nn.Sequential(*layers)

    def train(self, trial):
        
        #create model
        model = Model.architecture(self, trial).to(self.device)
        
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
        trial_file   = 'losses/loss_%d.txt'%(trial.number)
        model_file = 'models/model_%d.pt'%(trial.number)

        #split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.samples, self.features, train_size=0.7, shuffle=True)
        
        #cast data as torch tensor with correct shape and data type
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        
        for epoch in range(self.epochs):
            
            #????
            train_loss1, train_loss = torch.zeros(len(g)).to(device), 0.0
            train_loss2, points     = torch.zeros(len(g)).to(device), 0

            model.train()

            for X_train, y_train in zip(X_train, y_train):

                batch_size = len(X_train)

                X = X_train.to(device)
                y = y_train.to(device)
                p = model(X)

                #get posterior mean and error
                y_NN = p[:,g].squeeze()
                e_NN = p[:,h].squeeze()

                #get loss of mean and error
                loss1 = torch.mean((y_NN - y)**2, axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)

                #get total loss
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))

                #get training run losses overall
                train_loss1 += loss1*batch_size
                train_loss2 += loss2*batch_size

                points += batch_size

                #fancy machine learning stuff
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
            train_loss = torch.mean(train_loss).item()

            #now we move on to validation
            valid_loss1, valid_loss = torch.zeros(len(g)).to(device), 0.0
            valid_loss2, points     = torch.zeros(len(g)).to(device), 0

            model.eval()

            for X_train, y_train in zip(X_train, y_train):

                with torch.no_grad():

                    batch_size = len(x)

                    X = X_train.to(device)
                    y = y_train.to(device)
                    p = model(X)
                    
                    #get posterior mean and error
                    y_NN = p[:,g].squeeze()
                    e_NN = p[:,h].squeeze()

                    #get loss of mean and error
                    loss1 = torch.mean((y_NN - y)**2, axis=0)
                    loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)

                    #get total loss
                    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))

                    #get training run losses overall
                    valid_loss1 += loss1*batch_size
                    valid_loss2 += loss2*batch_size

                    points     += batch_size

                valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
                valid_loss = torch.mean(valid_loss).item()



                print('%03d %.3e %.3e '%(epoch, train_loss, valid_loss), end='')



                # save best model if found
                if valid_loss<min_valid:  

                    min_valid = valid_loss

                    torch.save(model.state_dict(), model_file)

                    print('(C) ', end='')

                print('')



                f = open(trial_file, 'a')
                f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
                f.close()


            return min_valid
