# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils.util import train_test_split, train_valid_split, data_loader, VanillaNN, aggregate_metrics


def main(args):

    t0 = time.time()

    ptd = args.ptd
    ptx = ptd + 'processed/'

    n_splits = args.n_splits
    n_ensembles = args.n_ensembles

    hidden_dims = args.hidden_dims
    experiment_name = args.experiment_name
    encoder_dims = args.encoder_dims
    decoder_dims = args.decoder_dims
    latent_dim = args.latent_dim
    log_name = '{}_log_test.txt'.format(experiment_name)
    args_name = '{}_args.txt'.format(experiment_name)
    pts = '../results/{}/'.format(experiment_name)

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
    (true_test, true_err_test, concs_test, lbs_test) = data_test

    f = open(pts + log_name, 'w')

    preds_test = []

    for n_split in range(n_splits):

        f.write('\nLoading data for split {}...'.format(n_split))
        f.flush()

        data_valid, data_train = train_valid_split(data_train_valid, n_split=n_split)
        (true_train, true_err_train, concs_train, lbs_train) = data_train

        # Load training and validation data
        X_train, ns_train, X_test, ns_test, mu_x, std_x = data_loader(concs_train, lbs_train, concs_test, lbs_test, ptx)

        # Scale y data
        sc_y = StandardScaler()
        y_train = torch.tensor(sc_y.fit_transform(true_train), dtype=torch.float32).reshape(-1, 1)

        preds_train = []

        for run_id in range(n_ensembles):
            # Load saved models
            encoder = VanillaNN(in_dim=mu_x.shape[0], out_dim=latent_dim, hidden_dims=encoder_dims)
            model = VanillaNN(in_dim=latent_dim, out_dim=1, hidden_dims=hidden_dims)
            encoder.load_state_dict(torch.load(pts + 'models/' + 'encoder{}{}_{}.pkl'.format(experiment_name,
                                                                                             n_split, run_id)))
            model.load_state_dict(torch.load(pts + 'models/' + 'model{}{}_{}.pkl'.format(experiment_name, n_split,
                                                                                         run_id)))
            encoder.eval()
            model.eval()

            Z_train = encoder(X_train).detach()
            Z_test = encoder(X_test).detach()
            # Make prediction
            pred_train = model.predict(Z_train, ns_train)
            pred_test = model.predict(Z_test, ns_test)
            pred_train = sc_y.inverse_transform(pred_train.detach().numpy())
            pred_test = sc_y.inverse_transform(pred_test.detach().numpy())
            preds_train.append(pred_train.reshape(-1))
            preds_test.append(pred_test.reshape(-1))

            rmse_train = np.sqrt(mean_squared_error(true_train, pred_train))
            rmse_test = np.sqrt(mean_squared_error(true_test, pred_test))
            r2_train = r2_score(true_train, pred_train)
            r2_test = r2_score(true_test, pred_test)

            print('\n Split {}, Run ID {}\tRMSE (Train): {:.4f}\tRMSE (Test): {:.4f}'.format(n_split, run_id,
                                                                                             rmse_train, rmse_test))
            f.write( '\n Split {}, Run ID {}\tRMSE (Train): {:.4f}\tRMSE (Test): {:.4f}\tR2 (Train): {:.4f}\tR2 (Test): {:.4f}'.format(n_split, run_id,
                    rmse_train, rmse_test, r2_train, r2_test))
            f.flush()

            np.save(pts + 'predictions/' + '{}{}_{}_pred_test.npy'.format(experiment_name, n_split, run_id),
                    pred_test)
            np.save(pts + 'predictions/' + '{}{}_{}_y_test.npy'.format(experiment_name, n_split, run_id),
                    true_test)
            np.save(pts + 'predictions/' + '{}{}_{}_y_err_test.npy'.format(experiment_name, n_split, run_id),
                    true_err_test)
            np.save(pts + 'predictions/' + '{}{}_{}_concs_test.npy'.format(experiment_name, n_split, run_id),
                    concs_test)
            np.save(pts + 'predictions/' + '{}{}_{}_lbs_test.npy'.format(experiment_name, n_split, run_id),
                    lbs_test)
        preds_train = np.vstack(preds_train)
        preds_train_mn = np.mean(preds_train, axis=0)
        preds_train_std = np.std(preds_train, axis=0)
        f.write('\nSplit {}\t Aggregate RMSE (train): {:.4f}\t R2 (train): {:.4f}'.format(n_split, np.sqrt(mean_squared_error(true_train, preds_train_mn)),
                                                                                          r2_score(true_train, preds_train_mn)))
        f.flush()
        np.save(pts + 'predictions/' + '{}{}_pred_train_mn.npy'.format(experiment_name, n_split),
                preds_train_mn)
        np.save(pts + 'predictions/' + '{}{}_pred_train_std.npy'.format(experiment_name, n_split),
                preds_train_std)
        np.save(pts + 'predictions/' + '{}{}_true_train_mn.npy'.format(experiment_name, n_split),
                true_train)
        np.save(pts + 'predictions/' + '{}{}_true_train_std.npy'.format(experiment_name, n_split),
                true_err_train)

    preds_test = np.vstack(preds_test)
    preds_test_mn = np.mean(preds_test, axis=0)
    preds_test_std = np.std(preds_test, axis=0)
    f.write('\nAggregate RMSE (test): {:.4f}\t R2 (test): {:.4f}'.format(np.sqrt(mean_squared_error(true_test, preds_test_mn)),
                                                     r2_score(true_test, preds_test_mn)))
    f.flush()
    np.save(pts + 'predictions/' + '{}pred_test_mn.npy'.format(experiment_name),
            preds_test_mn)
    np.save(pts + 'predictions/' + '{}pred_test_std.npy'.format(experiment_name),
            preds_test_std)
    np.save(pts + 'predictions/' + '{}true_test_mn.npy'.format(experiment_name),
            true_test)
    np.save(pts + 'predictions/' + '{}true_test_std.npy'.format(experiment_name),
            true_err_test)

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--experiment_name', type=str, default='NEW_VAE',
                        help='Name of experiment.')
    parser.add_argument('--hidden_dims', nargs='+', type=int,
                        default=[50, ],
                        help='Dimensionality of network hidden layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Number of systems to use in training.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency.')
    parser.add_argument('--n_ensembles', type=int, default=1,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')
    parser.add_argument('--encoder_dims', nargs='+', type=int,
                        default=[100, 100],
                        help='Dimensionality of encoder hidden layers.')
    parser.add_argument('--decoder_dims', nargs='+', type=int,
                        default=[100, 100],
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('--latent_dim', type=int, default=50,
                        help='Size of latent variable.')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Size of latent variable.')
    parser.add_argument('--ae_print_freq', type=int, default=50,
                        help='Print frequency for autoencoder training.')
    parser.add_argument('--ae_epochs', type=int, default=500,
                        help='Print frequency for autoencoder training.')

    args = parser.parse_args()

    main(args)
