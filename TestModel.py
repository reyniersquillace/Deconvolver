import torch
import numpy as np
import GenerateData
import argparse

def args():
        '''
        This function parses the command line input.

        Returns:
        --------
            training_data (str.npz): the unsplit training set
            outfile (str.pkl): the name of the file in which to save the model 
        '''
        parser = argparse.ArgumentParser(
                prog = "TestModel",
                description = "This progam evaluates a model saved as .pt over a number of test pulses.",
                epilog = "Shane u better like my fancy user interface",
                )
        parser.add_argument('model')
        parser.add_argument('n_test')
        args = parser.parse_args()
        return args.model, args.n_test

def test(model, n_test):
    '''This function tests the model on [n_test] fake pulses.

    Inputs:
    ------
        model (str): the name of the .pt file containing the PyTorch model
        n_test (int): the number of fake pulses on which to test the model
    '''
    m = torch.load(f'./torch_jar/{model}')
    m.eval()
    test_pulses, test_locs = GenerateData.generate_dummy(n_test, 1024)
    y_preds = []

    with torch.no_grad():
        for i in range(n_test):
            X_sample = test_pulses[i]
            X_sample = torch.tensor(X_sample, dtype=torch.float32)
            y_pred = m(X_sample)[0].item()
            y_pred_err = m(X_sample)[1].item()
            y_preds.append(y_pred)
            print(f"Predicted y: {y_pred} +/- {y_pred_err}")
            print(f"Actual y: {test_locs[i]}")
            print("\n")

model, n_test = args()
test(mode, n_test)
