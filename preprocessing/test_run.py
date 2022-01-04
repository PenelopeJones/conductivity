import os
import argparse
import time
import sys

import numpy as np
from ase import Atoms
import MDAnalysis as mda

import pdb

def create_mda(dcd_file, data_file): # loads trajectory with unwrapped coordinates
    u = mda.Universe(data_file, dcd_file)
    return u

def define_atom_types(u):
    # sort atoms into type of molecule
    anions = u.select_atoms("type Anion")
    cations = u.select_atoms("type Cation")
    solvent = u.select_atoms("type Solvent")
    return cations, anions, solvent

def create_position_arrays(u, anions, cations, solvent):
    # generate numpy arrays with all atom positions
    # position arrays: [time, ion index, spatial dimension (x/y/z)]
    time = 0
    n_times = u.trajectory.n_frames
    anion_positions = np.zeros((n_times, len(anions), 3))
    cation_positions = np.zeros((n_times, len(cations), 3))
    solvent_positions = np.zeros((n_times, len(solvent), 3))
    for ts in u.trajectory:
        anion_positions[time, :, :] = anions.positions - u.atoms.center_of_mass()
        cation_positions[time, :, :] = cations.positions - u.atoms.center_of_mass()
        solvent_positions[time, :, :] = solvent.positions - u.atoms.center_of_mass()
        time += 1
    return anion_positions, cation_positions, solvent_positions

def mda_to_numpy(conc, lb, ptd='../../../../../rds/hpc-work/conductivity/data/md-trajectories/'):
    dcd_file = '{}conc{}_lb{}.dcd'.format(ptd, conc, lb)
    data_file = '{}initial_config_conc{}.gsd'.format(ptd, conc)
    print(type(data_file))
    print(data_file)
    pdb.set_trace()
    u = create_mda(dcd_file, data_file)
    box_length = u.dimensions[0] # box length (use to wrap coordinates with periodic boundary conditions)
    cations, anions, solvent = define_atom_types(u)
    anion_positions, cation_positions, solvent_positions = (
                 create_position_arrays(u, anions, cations, solvent))

    return anion_positions, cation_positions, solvent_positions, box_length


def main(args):

    conc = args.conc
    lb = args.lb
    if conc == 0.001:
        ptd = args.ptd + '0001/'
    else:
        ptd = args.ptd + '{}/'.format(conc)
    pts = args.pts
    nt = 25000

    ptf = pts + 'X_{}_{}_soap'.format(conc, lb).replace('.', '-') + '.npy'

    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptd)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../../../../rds/hpc-work/conductivity/data/md-trajectories/',
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
