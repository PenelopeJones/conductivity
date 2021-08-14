import os
import argparse
import time
import sys

import numpy as np

from utils.feature_util import static_feature_vector
from utils.mda_util import mda_to_numpy

import pdb


def main(args):

    conc = args.conc
    lb = args.lb
    ptd = args.ptd + '{}/'.format(conc)
    pts = args.pts
    nt = 25000

    ptf = pts + 'X_{}_{}'.format(conc, lb).replace('.', '-') + '.npy'

    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptd)

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_cations = cation_positions.shape[1]
    n_solvents = solvent_positions.shape[1]

    x = []

    n_snaps = int(0.5*nt / n_cations) + 1 # number of snapshots needed to get dataset size > nt

    skip_snaps = n_snapshots // n_snaps
    print(skip_snaps)

    if not os.path.exists(args.pts):
        os.makedirs(args.pts)

    print('Concentration {}\t lB {}'.format(conc, lb))

    i = 0
    for snapshot_id in range(0, n_snapshots, skip_snaps):
        t0 = time.time()

        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        #solvents = solvent_positions[snapshot_id, :, :]

        for anion_id in range(n_anions):
            x.append(static_feature_vector(anions, cations, None, anion_id, box_length))
        for cation_id in range(n_cations):
            x.append(static_feature_vector(cations, anions, None, cation_id, box_length))
        np.save(ptf, np.vstack(x))
        if i % args.print_freq == 0:
            print('Snapshot {}\t Time: {:.1f}'.format(i, (time.time() - t0)))
        i+=1
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../data/processed/',
                        help='Path to directory where data is saved.')
    parser.add_argument('--conc', type=float, default=0.045,
                        help='Concentration.')
    parser.add_argument('--lb', type=float, default=10.0,
                        help='Bjerrum length.')
    parser.add_argument('--print_freq', type=int, default=2,
                        help='Print every N snapshots.')
    args = parser.parse_args()

    main(args)
