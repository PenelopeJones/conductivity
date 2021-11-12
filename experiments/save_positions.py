import os
import argparse

from preprocessing.utils.mda_util import mda_to_numpy

import pdb

import numpy as np

def correlation_function(anions, conductivities, min_r_value=0, max_r_value=4.0, bin_size=0.1, box_length=12.0):
    x = np.arange(min_r_value+0.5*bin_size, max_r_value+0.5*bin_size, bin_size)
    y = np.zeros(x.shape[0])
    n = np.zeros(x.shape[0])

    product = np.matmul(conductivities.reshape(-1, 1), conductivities.reshape(1, -1))
    distances = np.zeros(conductivities.shape)
    for i in range(anions.shape[0]):
        anion = anions[i, :].reshape(1, 3)
        distances[i, :] = np.linalg.norm(np.minimum(((anions - anion) % box_length), ((anion - anions) % box_length)),
                                         axis=1)
    product = product.reshape(-1)
    distances = distances.reshape(-1)
    for j in range(x.shape[0]):
        selected = product[np.where(np.abs(distances - x[j]) < 0.5*bin_size)].reshape(-1)
        y[j] += selected.sum()
        n[j] += selected.shape[0]
    return x, y, n

"""
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
"""


def main(args):

    conc = args.conc
    lb = args.lb
    experiment_name = args.experiment_name
    if conc == 0.001:
        ptd = args.ptd + '0001/'
    else:
        ptd = args.ptd + '{}/'.format(conc)
    nt = 25000
    n_splits = 5
    n_ensembles = 5
    min_r_value = 0
    max_r_value = 4.0
    bin_size = 0.2

    # Load ion positions
    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptd)

    # Load local conductivity predictions
    ptp = '../results/{}/'.format(experiment_name)
    preds = []
    for n_split in range(n_splits):
        for run_id in range(n_ensembles):
            pts_local = ptp + 'predictions/local_pred_{}_{}_{}_{}'.format(conc, lb, n_split, run_id).replace('.', '-') + '.npy'
            pred = np.load(pts_local)
            preds.append(pred)
    preds = np.vstack(preds)
    preds_mn = np.mean(preds, axis=0)
    preds_std = np.std(preds, axis=0)

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_snaps = int(nt / n_anions) + 1 # number of snapshots needed to get dataset size > nt
    skip_snaps = n_snapshots // n_snaps
    print(skip_snaps)

    if not os.path.exists(args.pts):
        os.makedirs(args.pts)

    if not os.path.exists(ptp + 'snapshots'):
        os.makedirs(ptp + 'snapshots')

    if not os.path.exists(ptp + 'figures'):
        os.makedirs(ptp + 'figures')

    print('Concentration {}\t lB {}'.format(conc, lb))

    idx = 0
    cfs = []
    nums = []

    for snapshot_id in range(0, n_snapshots, skip_snaps):
        print(snapshot_id)
        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        conductivities_mn = preds_mn[idx:(idx+anions.shape[0]), :]
        conductivities_std = preds_std[idx:(idx + anions.shape[0]), :]
        idx += anions.shape[0]
        np.save(ptp + 'snapshots/anions_{}_{}_{}.npy'.format(conc, lb, snapshot_id), anions)
        np.save(ptp + 'snapshots/cations_{}_{}_{}.npy'.format(conc, lb, snapshot_id), cations)
        np.save(ptp + 'snapshots/conductivity_mn_{}_{}_{}.npy'.format(conc, lb, snapshot_id), conductivities_mn)
        np.save(ptp + 'snapshots/conductivity_std_{}_{}_{}.npy'.format(conc, lb, snapshot_id), conductivities_std)

        if snapshot_id == 0:
            x, cf, num = correlation_function(anions, conductivities_mn, min_r_value=min_r_value,
                                              max_r_value=max_r_value, bin_size=bin_size,
                                              box_length=box_length)
        else:
            _, cf, num = correlation_function(anions, conductivities_mn, min_r_value=min_r_value,
                                              max_r_value=max_r_value, bin_size=bin_size,
                                              box_length=box_length)
        cfs.append(cf)
        nums.append(num)
    cfs = np.vstack(cfs)
    nums = np.vstack(nums)
    cfs = np.sum(cfs, axis=0)
    nums = np.sum(nums, axis=0)
    np.seterr(divide='ignore')
    cf = np.divide(cfs, nums)
    print(cf)
    np.save(ptp + 'bin_positions_{}_{}.npy'.format(conc, lb), x)
    np.save(ptp + 'correlation_function_{}_{}.npy'.format(conc, lb), cf)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.0)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    ax.scatter(x, cf, color='tab:blue', linewidth=2.0, alpha=0.7)
    ax.set_xlabel('Distance', fontsize=fontsize)
    ax.set_ylabel('Correlation function', fontsize=fontsize)
    ax.set_xlim(min_r_value, max_r_value)
    
    figsize = (7,7)
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Generate plot
        midpoint = 1- conductivities_mn.max()/(conductivities_mn.max() - conductivities_mn.min())
        orig_cmap = mpl.cm.coolwarm
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shifted')

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(anions[:, 0], anions[:, 1], anions[:, 2], marker='o', c=conductivities_mn, cmap=shifted_cmap)
        fig.savefig(ptp + 'figures/conductivity_{}_{}_{}.png'.format(conc, lb, snapshot_id), dpi=400)
    """

    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--conc', type=float, default=0.045,
                        help='Concentration.')
    parser.add_argument('--lb', type=float, default=10.0,
                        help='Bjerrum length.')
    parser.add_argument('--experiment_name', type=str, default='NEW_VAE_ENSEMBLE',
                        help='Name of experiment.')
    args = parser.parse_args()

    main(args)
