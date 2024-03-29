import os
import argparse
import time
import sys

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP, ACSF

from utils.feature_util import static_feature_vector
from utils.mda_util import mda_to_numpy

import pdb


def main(args):

    conc = args.conc
    lb = args.lb
    if conc == 0.001:
        ptd = args.ptd + '0001/'
    else:
        ptd = args.ptd + '{}/'.format(conc)
    pts = args.pts
    nt = 25000

    # SOAP descriptor parameters
    rcut = 5.0
    nmax = 6
    lmax = 5
    sparse = False

    anion_positions, cation_positions, _, box_length = mda_to_numpy(conc, lb, ptd)

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_cations = cation_positions.shape[1]

    nt = n_snapshots * n_anions
    n_splits = nt // 25000 + 1

    print(n_cations)

    cat_sym = 'Na'
    an_sym = 'Cl'
    species = [an_sym, cat_sym]
    symbols = [an_sym]*n_anions + [cat_sym]*n_cations
    soap_generator = SOAP(species=species, periodic=True,
                          rcut=rcut, nmax=nmax, lmax=lmax,
                          sparse=sparse)

    if not os.path.exists(args.pts):
        os.makedirs(args.pts)

    print('Concentration {}\t lB {}'.format(conc, lb))

    split = 0

    ptfa = pts + 'X_{}_{}_soap_anion_spatial_{}'.format(conc, lb, split).replace('.', '-') + '.npy'
    ptfc = pts + 'X_{}_{}_soap_cation_spatial_{}'.format(conc, lb, split).replace('.', '-') + '.npy'
    x_cations = []
    x_anions = []

    count = 0

    i = 0

    for snapshot_id in range(0, n_snapshots, 1):
        t0 = time.time()

        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        #solvents = solvent_positions[snapshot_id, :, :]
        positions = np.vstack([anions, cations])
        system = Atoms(symbols=symbols, positions=positions,
                       cell=[box_length, box_length, box_length],
                       pbc=True)
        soap = soap_generator.create(system, positions=list(range(0, n_anions)))
        x_anions.append(soap)

        # Select ion positions at a given snapshot
        anions = cation_positions[snapshot_id, :, :]
        cations = anion_positions[snapshot_id, :, :]
        positions = np.vstack([anions, cations])
        system = Atoms(symbols=symbols, positions=positions,
                       cell=[box_length, box_length, box_length],
                       pbc=True)
        soap = soap_generator.create(system, positions=list(range(0, n_anions)))
        x_cations.append(soap)

        count += n_anions

        if count >= 25000:
            np.save(ptfa, np.vstack(x_anions))
            np.save(ptfc, np.vstack(x_cations))
            x_anions = []
            x_cations = []

            count = 0
            split += 1
            ptfa = pts + 'X_{}_{}_soap_anion_spatial_{}'.format(conc, lb, split).replace('.', '-') + '.npy'
            ptfc = pts + 'X_{}_{}_soap_cation_spatial_{}'.format(conc, lb, split).replace('.', '-') + '.npy'

            print('Saved at snapshot {}\t Split {} Time: {:.1f}'.format(snapshot_id, split, (time.time() - t0)))

    print('All snapshots processed.')
    np.save(ptfa, np.vstack(x_anions))
    np.save(ptfc, np.vstack(x_cations))
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../../data/processed/',
                        help='Path to directory where data is saved.')
    parser.add_argument('--conc', type=float, default=0.045,
                        help='Concentration.')
    parser.add_argument('--lb', type=float, default=10.0,
                        help='Bjerrum length.')
    parser.add_argument('--print_freq', type=int, default=2,
                        help='Print every N snapshots.')
    args = parser.parse_args()

    main(args)
