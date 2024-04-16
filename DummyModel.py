import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from optuna.trial import Trial as trial

class Model(self):

    loss_fn = nn.MSELoss()  # mean square error
    #loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def __init__(self, input_size, output_size, max_layers = 3, max_neurons_layers = 500,
            device, epochs, seed, batch_size, workers, splits):

        self.input_size = input_size
        self.output_size = output_size
        self.max_layers = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.device = device
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.workers = workers
        self.splits = splits

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

        #load training data
        data = np.load('DummyData.npz')
        samples = data['pulses']
        features = data['locs']

