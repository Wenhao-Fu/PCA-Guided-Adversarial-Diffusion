from datetime import datetime
import numpy as np
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from utils import l2_loss, get_train_data, setup_logging
from modules import UNet_conditional
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from plot_field import plot_field

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        # Returns the cumulative product of elements of input in the dimension dim.
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        # self.alphas_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_ddpm(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)\
                    + torch.sqrt(beta) * noise
        model.train()
        return x

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def sample_ddim(self, model, batch_size, ddim_timesteps=100,
                    ddim_discr_method="uniform", ddim_eta=0.0, clip_denoised=True):
        logging.info(f"Sampling {batch_size} new images....")

        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.noise_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.noise_steps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.noise_steps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            ddim_timestep_seq = ddim_timestep_seq + 1
            # previous sequence
            ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
            # start from pure noise (for each example in the batch)
            x = torch.randn((batch_size, 1, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, ddim_timesteps)), position=0):

                # k
                t = torch.full((batch_size,), ddim_timestep_seq[i]).long().to(self.device)
                # s
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i]).long().to(self.device)

                # 1. get current and previous alpha_cumprod
                # alpha_k
                alpha_cumprod_t = self._extract(self.alpha_hat, t, x.shape)
                # alpha_s
                alpha_cumprod_t_prev = self._extract(self.alpha_hat, prev_t, x.shape)

                # 2. predict noise using model
                predicted_noise = model(x, t)

                # 3. get the predicted x_0
                pred_x0 = (x-torch.sqrt((1.-alpha_cumprod_t))*predicted_noise)/torch.sqrt(alpha_cumprod_t)

                if clip_denoised:
                    # Clamps all elements in input into the range
                    pred_x0 = torch.clamp(pred_x0, min=0., max=1.)

                # 4. compute variance: "sigma_t(η)"
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (
                                1 - alpha_cumprod_t / alpha_cumprod_t_prev))

                # 5. compute "direction pointing to x_t"
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * predicted_noise

                # 6. compute x_{t-1}
                # torch.randn_like Returns a tensor with the same size as input that is filled with random numbers
                # from a normal distribution with mean 0 and variance 1.
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)
        model.train()
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_train_data(args)
    model = UNet_conditional().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    l = len(dataloader)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("logsw", datetime.now().strftime('%Y%m%d_%H%M') +
                                        '_ndata{}'.format(args.training_sample_size)))

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, ) in enumerate(pbar):
            images = images.to(device).type(torch.cuda.FloatTensor)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, t)

            if args.mse:
                loss = mse(noise, predicted_noise)
            else:
                loss = l2_loss(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if (epoch + 1) % 50 == 0:
            field_gen = diffusion.sample_ddim(model, 10)
            field_gen = field_gen.cpu().numpy()
            plot_field(field_gen.reshape(10, -1), 'generation_perm'+str(epoch + 1)+'.jpg')

    torch.save(model.state_dict(), os.path.join("./models", args.run_name, f"ckpt.pt"))
    torch.save(optimizer.state_dict(), os.path.join("./models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 16
    args.epochs = 50
    args.image_size = 64
    args.training_sample_size = 2000
    args.device = "cuda"
    args.lr = 1e-4
    args.mse = True
    args.training_data_path = '../latent_perm.h5'
    args.run_name = 'Diffusion_unconditional_' + datetime.now().strftime('%Y%m%d_%H%M') + \
                    '_batchsize{}'.format(args.batch_size) + '_epochs{}'.format(args.epochs)
    train(args)


if __name__ == '__main__':
    launch()
