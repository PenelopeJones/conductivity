import os
import argparse
import time
import sys

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP, ACSF

from utils.mda_util_dumbbells import mda_to_numpy

import pdb

def main(args):

    conc = args.conc
    lb = args.lb
    frac = args.frac
    ptd = args.ptd
    pts = args.pts
    nt = 25000

    rcut = 5.0
    nmax = 6
    lmax = 5
    sparse = False

    ptf = pts + 'X_dumbbells_{}_{}_{}_soap'.format(frac, conc, lb).replace('.', '-') + '.npy'
    ptfl = pts + 'label_dumbbells_{}_{}_{}_soap'.format(frac, conc, lb).replace('.', '-') + '.npy'

    # extract positions of free anions, paired anions and all free/paired cations
    anion_free_positions, anion_paired_positions, anion_positions, _, _, cation_positions, _, box_length = mda_to_numpy_frac(conc, lb, frac, ptd)

    assert anion_positions.shape == cation_positions.shape

    (n_snapshots, n_anions, _) = anion_positions.shape
    n_cations = cation_positions.shape[1]
    n_anions_free = anion_free_positions.shape[1]
    n_anions_paired = anion_paired_positions.shape[1]
    print(n_cations)
    pdb.set_trace()

    cat_sym = 'Na'
    an_sym = 'Cl'
    species = [an_sym, cat_sym]
    symbols = [an_sym]*n_anions + [cat_sym]*n_cations

    soap_generator = SOAP(species=species, periodic=True,
                          rcut=rcut, nmax=nmax, lmax=lmax,
                          sparse=sparse)

    x = []

    n_snaps = int(nt / n_anions) # number of snapshots needed to get dataset size > nt
    skip_snaps = n_snapshots // n_snaps
    print(n_snaps)
    print(skip_snaps)

    if not os.path.exists(args.pts):
        os.makedirs(args.pts)

    print('Concentration {}\t lB {} Frac {}'.format(conc, lb, frac))

    i = 0
    label = np.array([1]*n_anions_free + [0]*n_anions_paired)
    labels = []

    for snapshot_id in range(0, n_snapshots, max(1, skip_snaps)):
        t0 = time.time()

        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        anions_free = anion_free_positions[snapshot_id, :, :]
        anions_paired = anion_paired_positions[snapshot_id, :, :]

        # solvents = solvent_positions[snapshot_id, :, :].
        # first n_anions_free positions are free anions, then n_anions_paired are paired anions etc.
        positions = np.vstack([anions_free, anions_paired, cations])

        system = Atoms(symbols=symbols, positions=positions,
                       cell=[box_length, box_length, box_length],
                       pbc=True)
        soap = soap_generator.create(system, positions=list(range(0, n_anions)))
        x.append(soap)
        labels.append(label)

        np.save(ptf, np.vstack(x))
        np.save(ptfl, np.hstack(labels))
        if i % args.print_freq == 0:
            print('Snapshot {}\t Time: {:.1f}'.format(i, (time.time() - t0)))
        i+=1
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/md-trajectories/220125_dumbbells/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../../data/processed/',
                        help='Path to directory where data is saved.')
    parser.add_argument('--conc', type=float, default=0.045,
                        help='Concentration.')
    parser.add_argument('--lb', type=float, default=10.0,
                        help='Bjerrum length.')
    parser.add_argument('--frac', type=float, default=0.1,
                        help='Fraction of ions in dumbbells.')
    parser.add_argument('--print_freq', type=int, default=2,
                        help='Print every N snapshots.')
    args = parser.parse_args()

    main(args)
