import os
import numpy as np
import gsd.hoomd

# Everything is in Lennard-Jones units, so the Boltzmann constant and temperature are both 1
kb = 1
T = 1

"""
Below is code to convert the HOOMD trajectory files into arrays containing all of the species' positions and velocities.
"""

# load trajectory file
f = gsd.hoomd.open(name='traj.gsd', mode='rb')
s = f[1]

# sort through species types using their charge
eps = 0.35475833333333334 # dielectric constant
cation_charge = 1/np.sqrt(eps)
anion_charge = -1/np.sqrt(eps)
cation_mask = np.isin(s.particles.charge,[cation_charge])
anion_mask = np.isin(s.particles.charge,[anion_charge])

cation_positions = []
cation_velocities = []
anion_positions = []
anion_velocities = []
for s in f: # loop over all frames
    cation_positions.append(s.particles.position[cation_mask])
    cation_velocities.append(s.particles.velocity[cation_mask])
    anion_positions.append(s.particles.position[anion_mask])
    anion_velocities.append(s.particles.velocity[anion_mask])
anion_velocities = np.array(anion_velocities)
anion_positions = np.array(anion_positions)
cation_velocities = np.array(cation_velocities)
cation_positions = np.array(cation_positions)

path = '../../data/md-trajectories/'
subdir = '6'

pts = path + 'positions/' + subdir + '/'
if not os.path.exists(pts):
    os.makedirs(pts)

np.save(pts + 'anion_positions.npy', anion_positions)
np.save(pts + 'cation_positions.npy', cation_positions)
np.save(pts + 'anion_velocities.npy', anion_velocities)
np.save(pts + 'cation_velocities.npy', cation_velocities)
