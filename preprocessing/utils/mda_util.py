import numpy as np
import MDAnalysis as mda

def create_mda(dcd_file, ptd='../../../../rds/hpc-work/conductivity/data/md-trajectories/'): # loads trajectory with unwrapped coordinates
    data_file = ptd + "initial_config.gsd"
    u = mda.Universe(data_file,dcd_file)
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

def mda_to_numpy(conc, lb, ptd='../../../../rds/hpc-work/conductivity/data/md-trajectories/'):
    dcd_file = '{}conc{}_lb{}.dcd'.format(ptd, conc,lb)
    u = create_mda(dcd_file)
    box_length = u.dimensions[0] # box length (use to wrap coordinates with periodic boundary conditions)
    cations, anions, solvent = define_atom_types(u)
    anion_positions, cation_positions, solvent_positions = (
                 create_position_arrays(u, anions, cations, solvent))

    return anion_positions, cation_positions, solvent_positions, box_length
