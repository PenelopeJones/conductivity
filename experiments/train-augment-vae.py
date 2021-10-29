# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils.util import train_test_split, train_valid_split, data_loader_augmented, VanillaNN, aggregate_metrics


def main(args):

    t0 = time.time()

    ptd = args.ptd
    ptx = ptd + 'processed/'

    n_splits = args.n_splits
    n_ensembles = args.n_ensembles
    n_augments = args.n_augments
    hidden_dims = args.hidden_dims
    run_id = args.run_id
    n_split = args.n_split
    experiment_name = args.experiment_name
    lr = args.lr
    epochs = args.epochs
    print_freq = args.print_freq
    encoder_dims = args.encoder_dims
    decoder_dims = args.decoder_dims
    latent_dim = args.latent_dim
    ae_epochs = args.ae_epochs
    ae_print_freq = args.ae_print_freq
    batch_size = args.batch_size
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
    # Load training and validation data
    X_train, ns_train, X_valid, ns_valid, mu_x, std_x = data_loader_augmented(concs_train, lbs_train, concs_valid,
                                                                              lbs_valid, ptx, n_augments=n_augments)
    n_train = X_train.shape[0]
    # Scale and augment y data
    sc_y = StandardScaler()
    y_train = torch.tensor(sc_y.fit_transform(true_train), dtype=torch.float32)
    y_train = torch.repeat_interleave(y_train, n_augments).reshape(-1, 1)

    # Augment y data
    true_train = np.repeat(true_train, n_augments)
    true_err_train = np.repeat(true_err_train, n_augments)
    true_valid = np.repeat(true_valid, n_augments)
    true_err_valid = np.repeat(true_err_valid, n_augments)
    concs_train = np.repeat(concs_train, n_augments)
    lbs_train = np.repeat(lbs_train, n_augments)
    concs_valid = np.repeat(concs_valid, n_augments)
    lbs_valid = np.repeat(lbs_valid, n_augments)

    f.write('\nLoaded data for split {}... Load Time: {:.1f}'.format(n_split, time.time() - t0))
    f.flush()

    # Auto encode
    encoder = VanillaNN(in_dim=mu_x.shape[0], out_dim=latent_dim, hidden_dims=encoder_dims)
    decoder = VanillaNN(in_dim=latent_dim, out_dim=mu_x.shape[0], hidden_dims=decoder_dims)

    ae_criterion = nn.MSELoss()
    ae_optimiser = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    t0 = time.time()
    ae_running_loss = 0

    # Pre-train using the full dataset
    for epoch in range(ae_epochs):
        ae_optimiser.zero_grad()

        idx = np.random.permutation(n_train)[0:batch_size]

        Z_tr = encoder(X_train[idx])
        X_rec = decoder(Z_tr)
        ae_loss = ae_criterion(X_rec, X_train[idx])

        ae_running_loss += ae_loss

        if epoch % ae_print_freq == 0:
            print('\nAE Epoch {}\tLoss: {}\tTrain Time: {:.1f}'.format(epoch, ae_running_loss / ae_print_freq,
                                                                       time.time()-t0))
            f.write('\nAE Epoch {}\tLoss: {}\tTrain Time: {:.1f}'.format(epoch, ae_running_loss / ae_print_freq,
                                                                         time.time()-t0))
            f.flush()
            ae_running_loss = 0
            t0 = time.time()

        ae_loss.backward()
        ae_optimiser.step()

    Z_train = encoder(X_train).detach()
    Z_valid = encoder(X_valid).detach()
    torch.save(encoder.state_dict(), pts + 'models/' + 'encoder{}{}_{}.pkl'.format(experiment_name, n_split, run_id))
    torch.save(decoder.state_dict(), pts + 'models/' + 'decoder{}{}_{}.pkl'.format(experiment_name, n_split, run_id))

    model = VanillaNN(in_dim=latent_dim, out_dim=1, hidden_dims=hidden_dims)
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    f.write('\nTraining... ')
    f.flush()

    running_loss = 0
    t0 = time.time()
    rmse_best = 1000000

    for epoch in range(epochs):
        optimiser.zero_grad()
        pred = model.predict(Z_train, ns_train)
        loss = criterion(pred, y_train)
        running_loss += loss

        if epoch % print_freq == 0:
            t1 = time.time()

            # Make prediction
            pred_train = model.predict(Z_train, ns_train)
            pred_valid = model.predict(Z_valid, ns_valid)
            pred_train = sc_y.inverse_transform(pred_train.detach().numpy())
            pred_valid = sc_y.inverse_transform(pred_valid.detach().numpy())

            f.write('\nEpoch {}\tLoss: {:.4f}\t Train Time: {:.1f}\t Predict Time: {:.1f}'.format(epoch, running_loss / print_freq,
                                                                                                  t1-t0, time.time()-t1))
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
    parser.add_argument('--run_id', type=int, default=0,
                        help='Run ID.')
    parser.add_argument('--n_ensembles', type=int, default=1,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_split', type=int, default=0,
                        help='Split number.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')
    parser.add_argument('--n_augments', type=int, default=10,
                        help='Factor by which to augment the training data.')
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
