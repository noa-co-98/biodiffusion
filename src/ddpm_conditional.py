# Import necessary libraries
import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext
import matplotlib.pyplot as plt
import os  # Add missing import statement

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
from IPython.display import display, HTML
from tqdm import tqdm
from utils import *
from modules.modules import UNet_conditional, EMA

torch.cuda.empty_cache()

# Define configuration using SimpleNamespace
config = SimpleNamespace(
    run_name="DDPM_conditional",
    epochs=100,#change to 100
    noise_steps=500,
    seed=42,
    batch_size=4,
    seq_size=400,
    num_classes=5,
    dataset_path='/kaggle/input/',
    train_folder="train",
    val_folder="test",
    device="cuda:0",
    slice_size=1,
    do_validation=True,
    fp16=True,
    log_every_epoch=10,
    num_workers=2,
    lr=5e-3)

# Set up logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02,
                 seq_size=400, num_classes=5, c_in=3, c_out=3, device="cuda:0", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.seq_size = seq_size
        self.c_in = c_in
        self.num_classes = num_classes

        # Use time_dim=128 to match the UNet's expected dimensions.
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, time_dim=128, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device

        self.mse = nn.MSELoss()


    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_signal(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new sensor samples....")
        model.eval()

        x = torch.randn((n, self.c_in, self.seq_size, 1)).to(self.device)
        for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
            t = (torch.ones(n) * i).long().to(self.device)
            predicted_noise = model(x, t, labels)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, t, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            alpha = alpha.expand(-1, self.c_in, self.seq_size, 1)
            alpha_hat = alpha_hat.expand(-1, self.c_in, self.seq_size, 1)
            beta = beta.expand(-1, self.c_in, self.seq_size, 1)
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

        # Denormalize using saved channel_means and channel_stds
        means_tensor = torch.tensor([-0.3485068807854082, 9.19893893178572, 1.6422336239112372],dtype=torch.float32).to(self.device).view(1, self.c_in, 1, 1)
        stds_tensor = torch.tensor([1.8018072468122244, 2.450260174331658, 3.6645135990321305],dtype=torch.float32).to(self.device).view(1, self.c_in, 1, 1)

        x = x * stds_tensor + means_tensor
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if not train:
            # For validation, disable gradients to save memory.
            self.model.eval()
            context = torch.inference_mode()
        else:
            self.model.train()
            context = torch.enable_grad()

        for i, batch in enumerate(self.train_dataloader):
            signals = batch['signal'].to(self.device)
            labels = batch.get('label', None)
            if labels is not None:
                labels = labels.to(self.device)
            t = self.sample_timesteps(signals.shape[0]).to(self.device)
            x_t, noise = self.noise_signal(signals, t)
            if np.random.random() < 0.1:
                labels = None
            with context, torch.autocast("cuda"):
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise.squeeze(-1), predicted_noise.squeeze(-1))
                avg_loss += loss
            if train:
                self.train_step(loss)

            # Delete intermediate tensors and clear cache
            del signals, labels, x_t, noise, predicted_noise, loss
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return (avg_loss / len(self.train_dataloader)).item()

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        print(f"Train dataloader type: {type(self.train_dataloader)}")
        print(f"Val dataloader type: {type(self.val_dataloader)}")
        batch = next(iter(self.train_dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch 'signal' shape: {batch['signal'].shape}")
        print(f"Batch 'label' shape: {batch['label'].shape if 'label' in batch else 'No label'}")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in range(args.epochs):
            logging.info(f"Starting epoch {epoch}")
            _ = self.one_epoch(train=True)
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                logging.info(f"Epoch {epoch} validation loss: {avg_loss}")
            torch.cuda.empty_cache()
        self.save_model(run_name=args.run_name, epoch=epoch)

    def save_model(self, run_name, epoch=-1, save_dir="/content/drive/MyDrive/my_model"):
        save_path = os.path.join(save_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(save_path, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, f"optim.pt"))

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

def parse_args(config):
    """
    Parses command line arguments and updates the configuration.

    Args:
        config: Configuration object.
    """
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.seq_size, help='sequence size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())

    # Update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    # Seed everything
    set_seed(config.seed)
    diffuser = Diffusion(config.noise_steps, seq_size=config.seq_size, num_classes=config.num_classes)
    diffuser.prepare(config)
    diffuser.fit(config)
    save_dir = "/content/drive/MyDrive/Colab Notebooks/my_model"
    diffuser.save_model(run_name="my_run", epoch=10, save_dir=save_dir)
