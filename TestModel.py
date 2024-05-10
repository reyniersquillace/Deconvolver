import pickle
import torch
import numpy as np
import GenerateData
import argparse
from sklearn.model_selection import train_test_split

def args():
    '''
    This function parses the command line input.

    Returns:
    --------
        model (str) : the name of the model file to be loaded
        n_test (int): the number of test pulses to evaluate
    '''
    parser = argparse.ArgumentParser(
            prog = "TestModel",
            description = "This progam evaluates a saved model over a number of test pulses.",
            epilog = "Shane u better like my fancy user interface",
            )
    parser.add_argument('model')
    parser.add_argument('n_test')
    args = parser.parse_args()
   
    return args.model, int(args.n_test)

def test(model, n_test):
    '''This function tests the model on [n_test] fake pulses.

    Inputs:
    ------
        model (str): the name of the .pt file containing the PyTorch model
        n_test (int): the number of fake pulses on which to test the model
    '''

    if model[-2:] == 'pt':
        m = torch.load(model)
        m.eval()
        test_pulses, test_locs = GenerateData.generate_dummy(n_test, 1024)

        with torch.no_grad():
            for i in range(n_test):
                X_sample = test_pulses[i]
                X_sample = torch.tensor(X_sample, dtype=torch.float32)
                y_pred = m(X_sample)[0].item()
                y_pred_err = m(X_sample)[1].item()
                print(f"Predicted y: {y_pred} +/- {y_pred_err}")
                print(f"Actual y: {test_locs[i]}")
                print("\n")
    
    elif model[-3:] == 'pkl':
        pulses, locs, gammas = GenerateData.generate(n_test, 1024)
        #X_train, X_test, y_train, y_test = train_test_split(pulses, locs, train_size = )
       
        with open(model, 'rb') as f:
            clf = pickle.load(f)

        for i in range(n_test):
            y_pred = np.where(clf.predict(pulses[i:i+1, :]) == 1)[1]
            y_act = np.where(locs[i:i+1, :] == 1)[1]
            print(f"Predicted y: {y_pred}")
            print(f"Actual y: {y_act}")
            print("\n")
    else:
        raise Exception('What on earth did you just ask me to load?')

model, n_test = args()
test(model, n_test)
