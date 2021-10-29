# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils.util import train_test_split, train_valid_split, data_loader, VanillaNN, aggregate_metrics


def main(args):

    t0 = time.time()

    ptd = args.ptd
    ptx = ptd + 'processed/'

    n_splits = args.n_splits
    n_ensembles = args.n_ensembles
    hidden_dims = args.hidden_dims
    run_id = args.run_id
    n_split = args.n_split
    experiment_name = args.experiment_name
    lr = args.lr
    epochs = args.epochs
    print_freq = args.print_freq
    log_name = '{}_log_{}_{}.txt'.format(experiment_name, n_split, run_id)
    args_name = '{}_args.txt'.format(experiment_name)
    pts = '../results/{}/'.format(experiment_name)

    # Load ion positions
    if (n_split == 0) and (run_id == 0):
        with open(pts + args_name, 'w') as f:
            f.write(str(args))

    y = np.load(ptd + 'molar_conductivities.npy')
    y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    y_err = y_err.reshape(-1, 1)
    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load(ptd + 'bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)

    data = (y, y_err, concs, lbs)
    data_train_valid, data_test = train_test_split(data, seed=10)
    data_valid, data_train = train_valid_split(data_train_valid, n_split=n_split)
    (true_train, true_err_train, concs_train, lbs_train) = data_train
    (true_valid, true_err_valid, concs_valid, lbs_valid) = data_valid
    print('n_split {}: Loaded data train / valid sets'.format(n_split))

    f = open(pts + log_name, 'w')
    f.write('\nLoading data for split {}...'.format(n_split))
    f.flush()

    # Load training and validation data
    X_train, ns_train, X_valid, ns_valid, mu_x, std_x = data_loader(concs_train, lbs_train, concs_valid, lbs_valid, ptx)

    # Scale y data
    sc_y = StandardScaler()
    y_train = torch.tensor(sc_y.fit_transform(true_train), dtype=torch.float32).reshape(-1, 1)

    f.write('\nLoaded data for split {}... Load Time: {:.1f}'.format(n_split, time.time() - t0))
    f.flush()

    t0 = time.time()

    model = VanillaNN(in_dim=mu_x.shape[0], out_dim=1, hidden_dims=hidden_dims)
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    f.write('\nTraining... Set Up Time: {:.1f}'.format(time.time() - t0))
    f.flush()

    for epoch in range(epochs):
        optimiser.zero_grad()
        pred = model.predict(X_train, ns_train)
        loss = criterion(pred, y_train)
        running_loss += loss

        if epoch % print_freq == 0:
            t1 = time.time()

            # Make prediction
            pred_train = model.predict(X_train, ns_train)
            pred_valid = model.predict(X_valid, ns_valid)
            pred_train = sc_y.inverse_transform(pred_train.detach().numpy())
            pred_valid = sc_y.inverse_transform(pred_valid.detach().numpy())

            f.write('\nEpoch {}\tLoss: {:.4f}\t Train Time: {:.1f}\t Predict Time: {:.1f}'.format(epoch, running_loss / print_freq, t1-t0, time.time()-t1))
            f.flush()
            running_loss = 0
            t0 = time.time()

            rmse_valid = np.sqrt(mean_squared_error(true_valid, pred_valid))

            if rmse_valid < rmse_best:
                f.write('\nSaving RMSE (valid): {}'.format(rmse_valid))
                f.flush()
                np.save(pts + 'predictions/' + '{}{}_{}_pred_train.npy'.format(experiment_name, n_split, run_id), pred_train)
                np.save(pts + 'predictions/' + '{}{}_{}_y_train.npy'.format(experiment_name, n_split, run_id), true_train)
                np.save(pts + 'predictions/' + '{}{}_{}_y_err_train.npy'.format(experiment_name, n_split, run_id), true_err_train)
                np.save(pts + 'predictions/' + '{}{}_{}_concs_train.npy'.format(experiment_name, n_split, run_id), concs_train)
                np.save(pts + 'predictions/' + '{}{}_{}_lbs_tr.npy'.format(experiment_name, n_split, run_id), lbs_train)
                np.save(pts + 'predictions/' + '{}{}_{}_pred_valid.npy'.format(experiment_name, n_split, run_id), pred_valid)
                np.save(pts + 'predictions/' + '{}{}_{}_y_valid.npy'.format(experiment_name, n_split, run_id), true_valid)
                np.save(pts + 'predictions/' + '{}{}_{}_y_err_valid.npy'.format(experiment_name, n_split, run_id), true_err_valid)
                np.save(pts + 'predictions/' + '{}{}_{}_concs_valid.npy'.format(experiment_name, n_split, run_id), concs_valid)
                np.save(pts + 'predictions/' + '{}{}_{}_lbs_valid.npy'.format(experiment_name, n_split, run_id), lbs_valid)
                rmse_best = rmse_valid

                torch.save(model.state_dict(), pts + 'models/' + 'model{}{}_{}.pkl'.format(experiment_name, n_split, run_id))

                if (n_split == 0) and (run_id == 0):
                    aggregate_metrics('train', f, pts, experiment_name, n_splits, n_ensembles)
                    aggregate_metrics('valid', f, pts, experiment_name, n_splits, n_ensembles)

        loss.backward()
        optimiser.step()

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--experiment_name', type=str, default='NO_VAE',
                        help='Name of experiment.')
    parser.add_argument('--hidden_dims', nargs='+', type=int,
                        default=[100, 100,],
                        help='Dimensionality of network hidden layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Number of systems to use in training.')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs.')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency.')
    parser.add_argument('--run_id', type=int, default=0,
                        help='Run ID.')
    parser.add_argument('--n_ensembles', type=int, default=1,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_split', type=int, default=0,
                        help='Split number.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')

    args = parser.parse_args()

    main(args)
