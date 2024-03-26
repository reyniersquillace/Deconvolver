import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

data = np.load('TrainingData.npz')
X = data['pulses']
output = np.array([data['locs'], data['gammas'], data['amps']])
y = np.reshape(output, (10000, 3, 3))

model = nn.Sequential(
    nn.Linear(1024, 10000),
    nn.ReLU(),
    nn.Linear(512, 5000),
    nn.ReLU(),
    nn.Linear(256, 2500),
    nn.ReLU(),
    nn.Linear(128, 1250),
    nn.ReLU(),
    nn.Linear(64, 625),
    nn.ReLU(),
    nn.Linear(32, 125),
    nn.ReLU(),
    nn.Linear(16, 25),
    nn.ReLU(),
    nn.Linear(3, 3)
)

loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)