import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from optuna.trial import Trial as trial

data = np.load('DummyData.npz')
X = data['pulses']
y = data['locs']

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
    
        layers = []
    
        n_layers = trial.suggest_int("n_layers", 1, self.max_layers)
    
        in_features = self.input_size

        for i in range(n_layers):
   
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, self.max_neurons_layers)

            layers.append(nn.Linear(in_features, out_features))

            layers.append(nn.LeakyReLU(0.2))

            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.8)

            layers.append(nn.Dropout(p))

            in_features = out_features
    
        # get the last layer
    
        layers.append(nn.Linear(out_features, output_size))

        # return the model

        return nn.Sequential(*layers)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    # scaler = StandardScaler()
    # scaler.fit(X_train_raw)
    # X_train = scaler.transform(X_train_raw)
    # X_test = scaler.transform(X_test_raw)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # training parameters
    n_epochs = 10   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    
    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    
    # training loop
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test.view(1, -1))
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
    
    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))

    return model
