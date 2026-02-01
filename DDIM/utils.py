import os
import torch
from torch.utils.data import DataLoader
from load_data import load_data


def l2_loss(pred, true):
    loss = torch.sum((pred-true)**2, dim=[1, 2, 3])
    return torch.mean(loss)


def get_train_data(args):
    perm = load_data(args)[:args.training_sample_size]
    perm = torch.as_tensor(perm.reshape(args.training_sample_size, 1, args.image_size, args.image_size))
    dataset = torch.utils.data.TensorDataset(perm)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
