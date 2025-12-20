import numpy as np
import h5py


def load_data(args):

    fname1 = args.training_data_path
    hf = h5py.File(fname1, 'r')
    phi = hf['phi'][:]
    mean_prior = (hf['mean_prior'][:]).reshape(-1, 1)
    X = hf['X'][:, :args.training_sample_size]
    perm = hf['perm'][:args.training_sample_size]
    hf.close()
    perm[perm == 0] = -1

    return phi, mean_prior, X, perm


