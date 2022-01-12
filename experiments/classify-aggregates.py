import os
import sys
sys.path.append('../../')
import argparse

from preprocessing.utils.mda_util import mda_to_numpy
from preprocessing.utils.feature_util import radial_distance

import pdb

import numpy as np

def classify_aggregate(anions, cations, r_cutoff, box_length):
    x = np.zeros((anions.shape[0], 2))
    for i in range(anions.shape[0]):
        x[i, 0] = is_paired(anions[i], cations, r_cutoff, box_length)
        x[i, 1] = is_paired(anions[i], anions, r_cutoff, box_length)
    return x

def is_paired(anion, cations, r_cutoff, box_length):
    paired = 0
    for i in range(cations.shape[0]):
        cation = cations[i, :]
        r = radial_distance(anion[0], anion[1], anion[2], cation[0], cation[1], cation[2], box_length)
        if np.abs(r - 0.5*r_cutoff) < 0.5*r_cutoff:
            paired += 1
    return paired


def main(args):

    conc = args.conc
    lb = args.lb
    experiment_name = args.experiment_name
    if conc == 0.001:
        ptd = args.ptd + '0001/'
    else:
        ptd = args.ptd + '{}/'.format(conc)
    nt = 25000
    r_cutoff = 1.6

    # Load ion positions
    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptd)

    # Load local conductivity predictions
    ptp = '../results/{}/'.format(experiment_name)

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_snaps = int(nt / n_anions) # number of snapshots needed to get dataset size > nt
    skip_snaps = n_snapshots // n_snaps
    print(n_snaps)
    print(skip_snaps)

    if not os.path.exists(ptp + 'predictions/paired/'):
        os.makedirs(ptp + 'predictions/paired/')

    print('Concentration {}\t lB {}'.format(conc, lb))
    aggregate_states = []

    for snapshot_id in range(0, n_snapshots, max(1, skip_snaps)):
        print(snapshot_id)
        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        aggregate_state = classify_aggregate(anions, cations, r_cutoff, box_length)
        charge_state = aggregate_state[:, 1] - aggregate_state[:, 0] + 1
        size_state = aggregate_state[:, 1] + aggregate_state[:, 0] + 1
        states, counts = np.unique(aggregate_state, return_counts=True)
        aggregate_states.append(aggregate_state)
        #print('Number paired: {} Percentage paired: {:.2f}'.format(np.sum(paired), 100*np.sum(paired) / paired.shape[0]))
        #paireds.append(paired.reshape(-1))
    aggregate_states = np.vstack(aggregate_states).reshape(-1, 2)
    states, counts = np.unique(aggregate_states, return_counts=True)
    print(states)
    print(counts)

    #print('Number paired: {} Percentage paired: {:.2f}'.format(np.sum(paireds), 100*np.sum(paireds) / paireds.shape[0]))
    np.save(ptp + 'predictions/paired/aggregate_states_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', aggregate_states)

    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--conc', type=float, default=0.045,
                        help='Concentration.')
    parser.add_argument('--lb', type=float, default=10.0,
                        help='Bjerrum length.')
    parser.add_argument('--experiment_name', type=str, default='220104_WL_ENSEMBLE',
                        help='Name of experiment.')
    args = parser.parse_args()

    main(args)
