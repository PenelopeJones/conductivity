import os
import argparse
import time
import sys
sys.path.append('../')

import numpy as np

import pdb


def main(args):
    ptd = '../../data/processed/'

    # Load ion positions
    y = np.load(ptd + 'molar_conductivities.npy')
    y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    y_err = y_err.reshape(-1, 1)
    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load(ptd + 'bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)

    for conc in concs:
        for lb in lbs:
            ptf = ptd + 'X_{}_{}_soap'.format(conc, lb).replace('.', '-') + '.npy'
            print('\nConc: {}\t lB: {}'.format(conc, lb))
            try:
                X = np.load(ptf, allow_pickle=True)
                print(X.shape)
            except:
                pdb.set_trace()
                print('Error in loading data...')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')

    args = parser.parse_args()

    main(args)
