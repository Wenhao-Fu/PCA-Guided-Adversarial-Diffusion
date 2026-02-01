import numpy as np
import h5py


def load_data(args):

    fname1 = args.training_data_path
    hf = h5py.File(fname1, 'r')
    perm = hf['perm'][:args.training_sample_size]
    hf.close()

    return perm

