import os
import argparse
import time
import sys
sys.path.append('../')

import numpy as np
from scipy.stats import norm, invwishart, multivariate_normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from collections import Counter

import pdb

import json
import logging
import os
import shutil

import torch

class Params():
    """
    Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

# Taken from CS230 https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py.
def load_checkpoint(checkpoint, model, optimiser=None):
    """
    Loads model parameters (state_dict) from file_path. If optimiser is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimiser: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimiser:
        optimiser.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def system_subsample(conc, lb, n_samples, ptd):
    """
    Sample a subset of the local environments for each system.
    """
    ptf = ptd + 'X_{}_{}'.format(conc, lb).replace('.', '-') + '.npy'
    x = np.load(ptf)
    return np.random.choice(x, size=n_samples, replace=False)

def sample_batch(concs, lbs, y, y_err, mu_x=None, std_x=None, n_systems=5, n_samples=5000, ptd='data/'):
    """
    1. Sample a subset (size n_systems) of the total set of systems.
    2. For each of those systems, sample a subset (size n_samples) of local environments for that system.
    """
    nt = concs.shape[0]
    pdb.set_trace()
    assert concs.shape[0] == lbs.shape[0] == y.shape[0] == y_err.shape[0]
    pdb.set_trace()
    ids = np.random.choice(nt, size=n_systems, replace=False)

    y_batch = torch.tensor(y[ids]).float()
    y_batch_err = torch.tensor(y_err[ids]).float()
    X_batch = []

    for i in range(n_systems):
        X_batch.append(system_subsample(concs[i], lbs[i], n_samples, ptd+'processed/'))
    pdb.set_trace()
    X_batch = np.vstack(X_batch)
    if mu_x is not None:
        X_batch = (X_batch - mu_x) / std_x
    X_batch = torch.tensor(X_batch).float()
    return X_batch, y_batch, y_batch_err

def train_test_split(y, y_err, concs, lbs, seed=seed, fraction_valid=0.1, fraction_test=0.1):
    """
    Split the set of systems into a training set, validation set, and test set.
    """
    np.random.seed(seed)
    n = concs.shape[0]
    idx = np.random.permutation(n)
    idx_te = idx[0:int(fraction_test*n)]
    idx_val = idx[int(fraction_test*n):(int(fraction_test*n)+int(fraction_valid*n))]
    idx_tr = idx[(int(fraction_test*n)+int(fraction_valid*n)):]

    data_test = (y[idx_te], y_err[idx_te], concs[idx_te], lbs[idx_te])
    data_val = (y[idx_val], y_err[idx_val], concs[idx_val], lbs[idx_val])
    data_train = (y[idx_tr], y_err[idx_tr], concs[idx_tr], lbs[idx_tr])

    return data_train, data_val, data_test

def x_scaler(concs, lbs, ptf):
    """
    Compute the mean and standard deviation of the set comprising all local
    environments of all systems in the training set.
    """
    X = []
    for i in range(concs.shape[0]):
        conc = concs[i]
        lb = lbs[i]
        ptf = ptd + 'X_{}_{}'.format(conc, lb).replace('.', '-') + '.npy'
        x = np.load(ptf)
        print('Conc {}\t lB {}:\t N = {}'.format(conc, lb, x.shape[0]))
        X.append(x)
    X = np.vstack(X)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    print(mu)
    print(std)
    return mu, std
