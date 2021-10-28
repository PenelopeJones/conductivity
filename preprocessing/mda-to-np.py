import os
import argparse

import numpy as np
import MDAnalysis as mda
import pdb

# Everything is in Lennard-Jones units, so the Boltzmann constant and temperature are both 1
kb = 1
T = 1
n = 1 # chain length = 1 here (normal, monomeric anions)

"""
Process trajectory files

Below is code to convert the LAMMPS trajectory files into an MDAnalysis "universe". MDAnalysis 
has some nice built-in data analysis featuers (e.g. for radial distribution functions), or we 
can just convert the trajectories into numpy arrays describing the positions of each type of 
atom over time (called anion_positions, cation_positions, and solvent_positions below).
"""

def create_mda(path): # loads trajectory with unwrapped coordinates
    # system.data files stores system topology (bonds, angles, etc.)
    data_file = path + "system.data"
    dcd_file = path + "traj_unwrapped.dcd"
    u = mda.Universe(data_file,dcd_file, format="LAMMPS")
    return u

def define_atom_types(u, n):
    # sort atoms into type of molecule
    anions = u.select_atoms("type 1")
    cations = u.select_atoms("type 2")
    solvent = u.select_atoms("type 3")
    return cations, anions, solvent

def generate_times(u, timestep=5, run_start=2):
    # create time array (data is collected every 50 \tau)
    times = []
    current_step = 0
    for ts in u.trajectory[run_start:]:
        times.append(current_step * timestep)
        current_step += 1
    times = np.array(times)
    return times

def create_position_arrays(n, u, anions, cations, solvent, times, run_start=2):
    # generate numpy arrays with all atom positions
    # position arrays: [time, ion index, spatial dimension (x/y/z)]
    time = 0
    anion_positions = np.zeros((len(times), len(anions), 3)) # np.array with dims [n_snapshots, n_anions, 3]
    cation_positions = np.zeros((len(times), len(cations), 3)) # np.array with dims [n_snapshots, n_cations, 3]
    solvent_positions = np.zeros((len(times), len(solvent), 3)) # np.array with dims [n_snapshots, n_solvent, 3]
    for ts in u.trajectory[run_start:]:
        anion_positions[time, :, :] = anions.positions - u.atoms.center_of_mass()
        cation_positions[time, :, :] = cations.positions - u.atoms.center_of_mass()
        solvent_positions[time, :, :] = solvent.positions - u.atoms.center_of_mass()
        time += 1
    return anion_positions, cation_positions, solvent_positions

# Main script

def main(args):

    u = create_mda(args.path + 'raw/' + args.subdir + '/')
    V = u.dimensions[0] ** 3  # volume
    times = generate_times(u)
    cations, anions, solvent = define_atom_types(u, n)
    anion_positions, cation_positions, _ = (create_position_arrays(n, u, anions, cations,
                                                                                   solvent, times))

    pts = args.path + 'positions/' + args.subdir + '/'
    if not os.path.exists(pts):
        os.makedirs(pts)

    np.save(pts + 'anion_positions.npy', anion_positions)
    np.save(pts + 'cation_positions.npy', cation_positions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--subdir', type=str, default='5',
                        help='Sub directory of interest.')

    args = parser.parse_args()

    main(args)