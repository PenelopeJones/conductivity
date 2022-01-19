import os
import sys

import numpy as np
import MDAnalysis as mda


def create_mda(dcd_file, data_file): # loads trajectory with unwrapped coordinates
    u = mda.Universe(data_file, dcd_file)
    return u

def check_files_exist(dcd_file, data_file):
    print('Trying to find files at')
    print(dcd_file)
    print(data_file)
    if os.path.isfile(dcd_file) and os.path.isfile(data_file):
        print('Located dcd and data files.')
    else:
        print('Did not locate dcd and data files.')
        sys.exit()
    return

def define_atom_types(u):
    # sort atoms into type of molecule
    anions_free = u.select_atoms("type Anion") # free anions
    anions_paired = u.select_atoms("type Anion_Pair") # anions in dumbbell
    anions = u.select_atoms("type Anion or type Anion_Pair") # all anions
    cations_free = u.select_atoms("type Cation") # free cations
    cations_paired = u.select_atoms("type Cation_Pair") # cations in dumbbell
    cations = u.select_atoms("type Cation or type Cation_Pair") # all anions
    solvent = u.select_atoms("type Solvent")
    return cations, anions, solvent, anions_free, anions_paired, cations_free, cations_paired

def create_position_arrays(u, anions_free, anions_paired, anions, cations_free, cations_paired, cations, solvent):
    # generate numpy arrays with all atom positions
    # position arrays: [time, ion index, spatial dimension (x/y/z)]
    time = 0
    n_times = u.trajectory.n_frames
    anion_paired_positions = np.zeros((n_times, len(anions_paired), 3))
    anion_free_positions = np.zeros((n_times, len(anions_free), 3))
    cation_paired_positions = np.zeros((n_times, len(cations_paired), 3))
    cation_free_positions = np.zeros((n_times, len(cations_free), 3))
    anion_positions = np.zeros((n_times, len(anions), 3))
    cation_positions = np.zeros((n_times, len(cations), 3))
    solvent_positions = np.zeros((n_times, len(solvent), 3))
    for ts in u.trajectory:
        anion_free_positions[time, :, :] = anions_free.positions - u.atoms.center_of_mass(pbc=True)
        anion_paired_positions[time, :, :] = anions_paired.positions - u.atoms.center_of_mass(pbc=True)
        anion_positions[time, :, :] = anions.positions - u.atoms.center_of_mass(pbc=True)
        cation_free_positions[time, :, :] = cations_free.positions - u.atoms.center_of_mass(pbc=True)
        cation_paired_positions[time, :, :] = cations_paired.positions - u.atoms.center_of_mass(pbc=True)
        cation_positions[time, :, :] = cations.positions - u.atoms.center_of_mass(pbc=True)
        solvent_positions[time, :, :] = solvent.positions - u.atoms.center_of_mass(pbc=True)
        time += 1
    return anion_free_positions, anion_paired_positions, anion_positions, cation_free_positions, cation_paired_positions, cation_positions, solvent_positions

def mda_to_numpy(conc, lb, ptd='../../data/md-trajectories/'):
    dcd_file = '{}conc{}_lb{}.dcd'.format(ptd, conc, lb)
    data_file = '{}initial_config_dumbbells_conc{}.gsd'.format(ptd, conc)
    check_files_exist(dcd_file, data_file)
    u = create_mda(dcd_file, data_file)
    box_length = u.dimensions[0] # box length (use to wrap coordinates with periodic boundary conditions)
    cations, anions, solvent, anions_free, anions_paired, cations_free, cations_paired = define_atom_types(u)
    anion_free_positions, anion_paired_positions, anion_positions, cation_free_positions, cation_paired_positions, cation_positions, solvent_positions = (create_position_arrays(u, anions_free, anions_paired, anions, cations_free, cations_paired, cations, solvent))

    return anion_free_positions, anion_paired_positions, anion_positions, cation_free_positions, cation_paired_positions, cation_positions, solvent_positions, box_length
