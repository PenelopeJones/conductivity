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

    ptf = pts + 'X_{}_{}_soap_cat_an'.format(conc, lb).replace('.', '-') + '.npy'

    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptd)

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_cations = cation_positions.shape[1]
    n_solvents = solvent_positions.shape[1]
    print(n_cations)

    cat_sym = 'Na'
    an_sym = 'Cl'

    soap_generator_cat = SOAP(species=[cat_sym, an_sym], periodic=True,
                          rcut=rcut, nmax=nmax, lmax=lmax,
                          sparse=sparse)
    soap_generator_an = SOAP(species=[an_sym, cat_sym], periodic=True,
                          rcut=rcut, nmax=nmax, lmax=lmax,
                          sparse=sparse)

    symbols_cat = [cat_sym]*n_cations+[an_sym]*n_anions
    symbols_an = [an_sym]*n_anions+[cat_sym]*n_cations
    x = []

    n_snaps = int(nt / n_cations) # number of snapshots needed to get dataset size > nt
    skip_snaps = n_snapshots // n_snaps
    print(n_snaps)
    print(skip_snaps)

    if not os.path.exists(args.pts):
        os.makedirs(args.pts)

    print('Concentration {}\t lB {}'.format(conc, lb))

    i = 0


    for snapshot_id in range(0, n_snapshots, max(1, skip_snaps)):
        t0 = time.time()

        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        #solvents = solvent_positions[snapshot_id, :, :]

        system_cat = Atoms(symbols=symbols_cat, positions=np.vstack([cations, anions]),
                       cell=[box_length, box_length, box_length],
                       pbc=True)
        soap_cat = soap_generator_cat.create(system_cat, positions=list(range(0, n_cations)))

        system_an = Atoms(symbols=symbols_an, positions=np.vstack([anions, cations]),
                       cell=[box_length, box_length, box_length],
                       pbc=True)
        soap_an = soap_generator_an.create(system_an, positions=list(range(0, n_anions)))

        x.append(soap_cat)
        x.append(soap_an)

        np.save(ptf, np.vstack(x))
        if i % args.print_freq == 0:
            print('Snapshot {}\t Time: {:.1f}'.format(i, (time.time() - t0)))
        i+=1
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
