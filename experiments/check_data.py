import os
import argparse
import time
import sys
sys.path.append('../')

import numpy as np

import pdb


def main(args):
    ptd = '../data/'

    # Load ion positions
    y = np.load(ptd + 'molar_conductivities.npy')
    y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    y_err = y_err.reshape(-1, 1)
    concs = np.load(ptd + 'concentrations.npy')
    lbs = np.load(ptd + 'bjerrum_lengths.npy')

    errors = []
    for conc in concs:
        for lb in lbs:
            ptf = ptd + 'processed/X_{}_{}_soap'.format(conc, lb).replace('.', '-') + '.npy'

            try:
                X = np.load(ptf, allow_pickle=True)
                print('\nConc: {}\t lB: {}\tN:{}, Dim:{}'.format(conc, lb, X.shape[0], X.shape[1]))
                print(X.shape)
            except:
                errors.append('Conc: {}\t lB: {}\t\n'.format(conc, lb))
    print(errors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')

    args = parser.parse_args()

    main(args)
