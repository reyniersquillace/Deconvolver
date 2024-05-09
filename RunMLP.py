import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
import numpy as np
import argparse
import Princeton

class MLPModel:
    
    def __init__(self):

        training_data, outfile = MLPModel.args()
        pulses, locs, gammas = MLPModel.get_data(training_data)
        self.pulses = pulses
        self.locs = locs
        self.gammas = gammas
        self.outfile = outfile
        self.clf = MLPModel.build(self)

    def args():
        '''
        This function parses the command line input.

        Returns:
        --------
            training_data (str.npz): the unsplit training set
            outfile (str.pkl): the name of the file in which to save the model 
        '''
        parser = argparse.ArgumentParser(
                prog = "RunMLP",
                description = "This progam trains a simple MLP classifier to identify peaks.",
                epilog = "Shane u better like my fancy user interface",
                )
        parser.add_argument('training_data')
        parser.add_argument('outfile')
        args = parser.parse_args()
        return args.training_data, args.outfile

    def get_data(training_data):
        '''
        This function loads the unsplit training set.

        Inputs: 
        -------
            training_data (npz): the npz file location with the training set

        Returns:
        --------
            pulses (arr): the pulse profiles of each sample
            locs (arr)  : a series of 1D arrays containing either 0s (no peak) or 1s (peak)
            gammas (arr): a series of integers with the FWHM of each subpulse
        '''

        data = np.load(training_data)
        pulses = np.zeros(np.shape(data['pulses']))
        locs = np.zeros(np.shape(data['locs']))

        for i in range(len(pulses)):
            pulses[i, :], locs[i, :] = Princeton.pseudo_princeton(pulses[i, :], locs[i, :])

        return pulses, locs, data['gammas']

    def build(self):
        '''
        This function calls the MLP classifier and trains it.

        Inputs:
        -------
            pulses (arr): the pulse profiles of each sample
            locs (arr)  : a series of 1D arrays containing either 0s (no peak) or 1s (peak)
        
        Returns:
        -------
            clf (MLP object): a trained MLP classifier model        
        '''
        X_train, X_test, y_train, y_test = train_test_split(self.pulses, self.locs, train_size=0.8)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        clf = MLP((1024, 1024), max_iter = 300).fit(X_train, y_train)
        
        return clf

    def stats(self):
        '''
        This function calculates statistics regarding the MLP model.

        Returns:
        --------
        
        stats_dict (dict): dictionary of statistical parameters
        '''

        overfit = 0
        underfit = 0
        correct = 0

        for i in range(len(self.X_test)):
            true = np.where(self.y_test[i:i+1, :] == 1)[1]
            pred = np.where(self.clf.predict(self.X_test[i:i+1, :]) == 1)[1]
            if len(true) > len(pred):
                underfit += 1
            elif len(pred) > len(true):
                overfit += 1
            else:
                correct += 1
    
        stats_dict = {}
        stats_dict['underfit'] = underfit/(underfit + overfit + correct)
        stats_dict['overfit'] = overfit/(underfit + overfit + correct)
        stats_dict['correct'] = correct/(underfit + overfit + correct)

        return stats_dict

    def save(self):
        '''
        This function saves the model as a pickle file.
        '''

        with open(self.outfile, 'wb') as f:
            pickle.dump(self.clf, f)

model = MLPModel()
stats = model.stats()
for stat in stats:
    print(f"{stat} = {stats[stat]}%")
model.save()
