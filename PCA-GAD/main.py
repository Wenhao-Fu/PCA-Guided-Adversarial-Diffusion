from datetime import datetime
import argparse
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils import l2_loss, get_train_data, setup_logging
from modules import Discriminator, Generator
import logging
from torch.utils.tensorboard import SummaryWriter
from plot_field import plot_field

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Gen_sampling:
    def __init__(self, device="cuda"):
        self.device = device

    def sample(self, model, n, mean_prior, phi):
        logging.info(f"Sampling {n} new images....")
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 200)).to(self.device)
            images = model(x, mean_prior, phi)
        model.train()
        return images


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader, phi, mean_prior = get_train_data(args)

    phi = phi.to(device).type(torch.cuda.FloatTensor)
    mean_prior = mean_prior.to(device).type(torch.cuda.FloatTensor)

    D = Discriminator().to(device)
    G = Generator().to(device)

    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr_d)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr_g)

    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

    l = len(dataloader)
    Gen_pre = Gen_sampling(device=device)
    logger = SummaryWriter(os.path.join("logsw", datetime.now().strftime('%Y%m%d_%H%M')
                                        + '_ndata{}'.format(args.training_sample_size)))

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, ) in enumerate(pbar):
            images = images.to(device).type(torch.cuda.FloatTensor)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            d_out_real, dr1, dr2 = D(images)

            if args.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif args.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            z = torch.randn(args.batch_size, 200).to(device)
            fake_images = G(z, mean_prior, phi)
            d_out_fake, df1, df2 = D(fake_images)

            if args.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif args.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            reset_grad()
            d_loss.backward()
            d_optimizer.step()

            if args.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(images.size(0), 1, 1, 1).cuda().expand_as(images)
                interpolated = Variable(alpha * images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out, _, _ = D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = args.lambda_gp * d_loss_gp

                reset_grad()
                d_loss.backward()
                d_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(args.batch_size, 200).to(device)
            fake_images = G(z, mean_prior, phi)

            # Compute loss with fake images
            g_out_fake, _, _ = D(fake_images)
            if args.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif args.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            reset_grad()
            g_loss_fake.backward()
            g_optimizer.step()

            pbar.set_postfix(d_loss=d_loss.item(), g_loss_fake=-g_loss_fake.item(),
                             d_loss_real=-d_loss_real.item(),
                             d_loss_fake=d_loss_fake.item())
            logger.add_scalar("d_loss", d_loss.item(), global_step=epoch * l + i)

        if (epoch + 1) % 10 == 0:
            field_gen = Gen_pre.sample(G, 10, mean_prior, phi)
            field_gen = field_gen.cpu().numpy()
            field_gen = (field_gen + 1) / 2
            plot_field(field_gen.reshape(10, -1), 'generation_perm'+str(epoch + 1)+'.jpg')

    torch.save(G.state_dict(), os.path.join("./models", args.run_name, f"generator_ckpt.pt"))
    torch.save(D.state_dict(), os.path.join("./models", args.run_name, f"discriminator_ckpt.pt"))


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 16
    args.epochs = 10
    args.image_size = 64
    args.training_sample_size = 2000
    args.device = "cuda"
    args.lr_g = 1e-4
    args.lr_d = 4e-4
    args.adv_loss = 'hinge'
    args.lambda_gp = 10
    args.training_data_path = '../latent_perm.h5'
    args.run_name = 'PCA_GAD_' + datetime.now().strftime('%Y%m%d_%H%M') + \
                    '_batchsize{}'.format(args.batch_size) + '_epochs{}'.format(args.epochs)
    train(args)


if __name__ == '__main__':
    launch()
