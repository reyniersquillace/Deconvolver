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

        and either:
        pulse (arr) : the array to be evaluated
        n_test (int): the number of test pulses to evaluate
    '''
    parser = argparse.ArgumentParser(
            prog = "CallModel",
            description = "This progam evaluates a saved model over a number of pulses.",
            epilog = "Shane u better like my fancy user interface",
            )
    parser.add_argument('model')
    parser.add_argument('setting') 
    parser.add_argument('third_arg')
    args = parser.parse_args()
    if args.setting == 'use':
        pulse_data = args.third_arg
        pulse = np.load(pulse_data)
        return args.model, pulse, args.setting
    elif args.setting == 'test':
        n_test = int(args.third_arg)
        return args.model, n_test, args.setting
    else:
        raise Exception('That\'s not a valid call type.')

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
        
        with open(model, 'rb') as f:
            clf = pickle.load(f)

        if model[13:17] == 'Mult':
            for i in range(n_test):
                y_pred = np.where(clf.predict([pulses[i]]) == 1)[1]
                y_act = np.where(locs[i:i+1, :] == 1)[1]
                print(f"Predicted y: {y_pred}")
                print(f"Actual y: {y_act}")
                print("\n")

        elif model[13:17] == 'Gamm':
            for i in range(n_test):
                y_pred = clf.predict([pulses[i]])[0, :5]
                y_act = gammas[i, :5]
                print(f"Predicted y: {y_pred}")
                print(f"Actual y: {y_act}")
                print("\n")
    else:
        raise Exception('What on earth did you just ask me to load?')

def use(model, pulse):
    '''
    This function allows the user to evalute an input array.

    Inputs:
    ------
        model (str): the name of the file containing the model
        pulse (arr): a pulse profile
    '''
    if model[-2:] == 'pt':
        m = torch.load(model)
        m.eval()

        with torch.no_grad():
            X_sample = torch.tensor(pulse, dtype=torch.float32)
            y_pred = m(X_sample)[0].item()
            y_pred_err = m(X_sample)[1].item()
            print(f"Predicted y: {y_pred} +/- {y_pred_err}")

    elif model[-3:] == 'pkl':
        
        with open(model, 'rb') as f:
            clf = pickle.load(f)
            y_pred = np.where(clf.predict([pulse]) == 1)[1]
            print(f"Predicted y: {y_pred}")
    else:
        raise Exception('What on earth did you just ask me to load?')

model, arg, call = args()
if call == 'test':
    test(model, arg)
else:
    use(model, arg)
