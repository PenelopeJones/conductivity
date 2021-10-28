import os
import argparse
import time
import sys
sys.path.append('../')

import numpy as np
from scipy.stats import norm, invwishart, multivariate_normal

from utils.feature_util import radial_distance, dynamic_feature_vector

import pdb


def main(args):
    ptl = args.path + 'positions/' + args.subdir + '/'

    # Load ion positions
    anion_positions = np.load(ptl + 'anion_positions.npy')
    cation_positions = np.load(ptl + 'cation_positions.npy')
    anion_velocities = np.load(ptl + 'anion_velocities.npy')
    cation_velocities = np.load(ptl + 'cation_velocities.npy')

    """
    sigma = 2.5A / 0.97
    cutoff is 1.6*sigma
    residence time of dynamics ion pairs determined by classifying ions as neighbours
    if they fall within 2.5*sigma of each other.
    residence time of static ion pairs classified as ions within 1.6*sigma of each other.
    """

    sigma = 1
    r_c_static = 1.6 * sigma
    r_c_dynamic = 2.5 * sigma
    box_length = 50000.0 ** (1 / 3) * sigma

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_cations = cation_positions.shape[1]

    # Identify which ions are classified as being in a static ion pair at each snapshot
    """
    x_stat = []
    x_dyn = []
    x_none = []

    for snapshot_id in range(0, args.n_snapshots, 5):

        t0 = time.time()
        # Select ion positions at a given snapshot
        anion_pos = anion_positions[snapshot_id, :, :]
        cation_pos = cation_positions[snapshot_id, :, :]
        anion_vel = anion_velocities[snapshot_id, :, :]
        cation_vel = cation_velocities[snapshot_id, :, :]

        x_cat = []
        x_an = []

        for anion_id in range(n_anions):
            x_an.append(dynamic_feature_vector(anion_pos, cation_pos, anion_vel, cation_vel, anion_id, box_length))

        for cation_id in range(n_cations):
            x_cat.append(dynamic_feature_vector(cation_pos, anion_pos, cation_vel, anion_vel, cation_id, box_length))


        for anion_id in range(n_anions):

            selected_ids = np.random.permutation(n_cations)[0:4]
            for cation_id in range(n_cations):
                r = radial_distance(anion_pos[anion_id, 0], anion_pos[anion_id, 1], anion_pos[anion_id, 2],
                                    cation_pos[cation_id, 0], cation_pos[cation_id, 1], cation_pos[cation_id, 2],
                                    box_length)
                if r <= r_c_dynamic:
                    if r <= r_c_static:
                        x_stat.append(np.concatenate((x_an[anion_id], x_cat[cation_id])).reshape(-1))

                    else:
                        x_dyn.append(np.concatenate((x_an[anion_id], x_cat[cation_id])).reshape(-1))

                else:
                    if cation_id in selected_ids:
                        x_none.append(np.concatenate((x_an[anion_id], x_cat[cation_id])).reshape(-1))
                    else:
                        continue
        if snapshot_id % args.print_freq == 0:
            print('Snapshot {}\t Time: {:.2f}'.format(snapshot_id, (time.time() - t0)))

    x_stat = np.vstack(x_stat)
    x_dyn = np.vstack(x_dyn)
    x_none = np.vstack(x_none)

    pdb.set_trace()

    if not os.path.exists(args.pts):
        os.makedirs(args.pts)

    np.save(args.pts + 'X_{}_stat.npy'.format(args.subdir), x_stat)
    np.save(args.pts + 'X_{}_dyn.npy'.format(args.subdir), x_dyn)
    np.save(args.pts + 'X_{}_none.npy'.format(args.subdir), x_none)
    """
    x_stat = np.load(args.pts + 'X_{}_stat.npy'.format(args.subdir))
    x_dyn = np.load(args.pts + 'X_{}_dyn.npy'.format(args.subdir))
    x_none = np.load(args.pts + 'X_{}_stat.npy'.format(args.subdir))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../data/processed/',
                        help='Path to directory containing data.')
    parser.add_argument('--subdir', type=str, default='6',
                        help='Sub directory of interest.')
    parser.add_argument('--print_freq', type=int, default=1,
                        help='Print every N snapshots.')
    parser.add_argument('--n_snapshots', type=int, default=25,
                        help='Number of snapshots.')
    parser.add_argument('--cutoff', type=int, default=150,
                        help='Expected maximum number of snapshots over which ions are paired.')

    args = parser.parse_args()

    main(args)
