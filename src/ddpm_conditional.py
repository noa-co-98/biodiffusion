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
    epochs=100,
    noise_steps=1000,
    seed=42,
    batch_size=10,
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
    num_workers=10,
    lr=5e-3)

# Set up logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


# Define the Diffusion class
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, seq_size=2*1024, num_classes=5, c_in=3, c_out=3,
                 device="cuda:0", **kwargs):
        """
        Initializes the Diffusion class.

        Args:
            noise_steps (int): Number of noise steps.
            beta_start (float): Starting value for beta in noise schedule.
            beta_end (float): Ending value for beta in noise schedule.
            seq_size (int): Sequence size.
            num_classes (int): Number of classes.
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            device (str): Device for computation.
            **kwargs: Additional keyword arguments.
        """
        # Initialize parameters
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        print(device)
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.seq_size = seq_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        """
        Prepares the noise schedule using beta_start and beta_end.

        Returns:
            torch.Tensor: Noise schedule.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        """
        Samples random timesteps.

        Args:
            n (int): Number of timesteps to sample.

        Returns:
            torch.Tensor: Sampled timesteps.
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_signal(self, x, t):
        """
        Adds noise to images at a specific timestep.

        Args:
            x (torch.Tensor): Input 3D signals.
            t (torch.Tensor): Timestep.

        Returns:
            tuple: Tuple containing noisy images and noise.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        """
        Generates samples using the diffusion model.

        Args:
            use_ema (bool): Whether to use the EMA model.
            labels (torch.Tensor): Labels for conditional sampling.
            cfg_scale (int): Configuration scale.

        Returns:
            torch.Tensor: Generated samples.
        """
        means = np.array([-0.07471468,  9.2097996,   1.67035854])
        stds = np.array([1.04284218, 1.61164589, 3.29274985])
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.seq_size, 1)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]  # Shape: [n, channels, 1, 1]
                alpha_hat = self.alpha_hat[t][:, None, None, None]  # Shape: [n, channels, 1, 1]
                beta = self.beta[t][:, None, None, None]  # Shape: [n, channels, 1, 1]
                predicted_noise = predicted_noise[:, :, :, None]
                # Ensure that alpha, alpha_hat, and beta are correctly reshaped
                alpha = alpha.expand(-1, self.c_in, self.seq_size, 1)  # Adjust to [batch_size, channels, width, height]
                alpha_hat = alpha_hat.expand(-1, self.c_in, self.seq_size, 1)
                beta = beta.expand(-1, self.c_in, self.seq_size, 1)
                print(f"x shape: {x.shape}")
                print(f"alpha shape: {alpha.shape}")
                print(f"alpha_hat shape: {alpha_hat.shape}")
                print(f"beta shape: {beta.shape}")

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                print(f"predicted_noise shape: {predicted_noise.shape}")
                print(f"noise shape: {noise.shape}")

                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        
        means_tensor = torch.tensor(means).to(self.device).view(1, self.c_in, 1, 1)  # Reshape for broadcasting
        stds_tensor = torch.tensor(stds).to(self.device).view(1, self.c_in, 1, 1)    # Reshape for broadcasting
        x = x * stds_tensor + means_tensor  # Denormalize
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        x = (x + 1) * 0.5
        print("good")
        return x

    def train_step(self, loss):
        """
        Performs a training step.

        Args:
            loss (torch.Tensor): Loss value.
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        """
        Performs one epoch of training or validation.

        Args:
            train (bool): Whether to perform training.

        Returns:
            float: Average loss for the epoch.
        """
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()

        print(f"train_dataloader type in one_epoch: {type(self.train_dataloader)}")
        print(f"val_dataloader type in one_epoch: {type(self.val_dataloader)}")
        batch = next(iter(self.train_dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch 'signal' shape: {batch['signal'].shape}")
        print(f"Batch 'label' shape: {batch['label'].shape if 'label' in batch else 'No label'}")
        print(f"Batch 'signal' type: {type(batch['signal'])}")
        print(f"Batch 'label' type: {type(batch['label']) if 'label' in batch else 'No label'}")
        pbar = self.train_dataloader

        for i, batch in enumerate(self.train_dataloader):
            signals = batch['signal']
            labels = batch['label'] if 'label' in batch else None
            print(f"Batch {i} - Signals type: {type(signals)}, Labels type: {type(labels)}")
            print(
                f"Batch {i} - Signals shape: {signals.shape}, Labels shape: {labels.shape if labels is not None else 'No labels'}")

            print(f"Batch {i} - Signals type: {type(signals)}, Labels type: {type(labels)}")
            print(
                f"Batch {i} - Signals shape: {signals.shape}, Labels shape: {labels.shape if labels is not None else 'No labels'}")

            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                print(signals)
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(signals.shape[0]).to(self.device)
                x_t, noise = self.noise_signal(signals, t)

                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)

                noise = noise.squeeze(-1)  # Adjust shape to match the target
                predicted_noise = predicted_noise.squeeze(-1)  # Adjust shape to match the target
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss

            if train:
                self.train_step(loss)

            pbar.comment = f"MSE={loss.item():2.3f}"
        return avg_loss.mean().item()

    def save_signals_to_drive(self, signals, labels, save_dir, prefix="sampled"):
        """
        Saves sampled signals to disk as .npy files.

        Args:
            signals (torch.Tensor): The signals to save.
            labels (torch.Tensor): The labels for the signals.
            save_dir (str): Directory to save the files.
            prefix (str): Prefix for file names.
        """
        # Ensure save_dir exists
        os.makedirs(save_dir, exist_ok=True)

        # Save each signal as a .npy file
        for i, (signal, label) in enumerate(zip(signals, labels)):
            # Convert tensor to numpy array
            signal_np = signal.cpu().numpy()  # Convert to numpy array

            # Construct file name and save
            file_name = f"{prefix}_label_{label.item()}.npy"
            file_path = os.path.join(save_dir, file_name)
            np.save(file_path, signal_np)

            print(f"Saved {file_path}")

    def plot_signals(self, signals, labels, save_dir, prefix="sampled"):
        """
        Plots and saves the sampled 3D signals as images.

        Args:
            signals (torch.Tensor): The signals to plot.
            labels (torch.Tensor): The labels for the signals.
            save_dir (str): Directory to save the plot files.
            prefix (str): Prefix for file names.
        """
        # Ensure save_dir exists
        os.makedirs(save_dir, exist_ok=True)

        num_channels = signals.shape[1]  # Number of channels

        # Plot each signal
        for i, (signal, label) in enumerate(zip(signals, labels)):
            plt.figure(figsize=(12, 4 * num_channels))
            for ch in range(num_channels):
                plt.subplot(num_channels, 1, ch + 1)
                plt.plot(signal[ch].cpu().numpy().squeeze(), label=f'Channel {ch}')
                plt.title(f'{prefix} Signal - Label {label.item()} - Channel {ch}')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.legend()

            # Save plot
            file_name = f"{prefix}_label_{label.item()}.png"
            file_path = os.path.join(save_dir, file_name)
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()  # Close the figure to free up memory

            print(f"Saved plot {file_path}")

    def log_signals(self, save_dir="/content/drive/MyDrive/Colab Notebooks/sampled_signals"):
        """
        Logs sampled signals to WandB and saves them to disk.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Sample signals
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_signals = self.sample(use_ema=False, labels=labels)
        ema_sampled_signals = self.sample(use_ema=True, labels=labels)

        # Save sampled signals
        self.save_signals_to_drive(sampled_signals, labels, save_dir, prefix="sampled")
        self.save_signals_to_drive(ema_sampled_signals, labels, save_dir, prefix="ema_sampled")

        # Plot sampled signals
        self.plot_signals(sampled_signals, labels, save_dir, prefix="sampled")
        self.plot_signals(ema_sampled_signals, labels, save_dir, prefix="ema_sampled")

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        """
        Loads model weights from checkpoint files.

        Args:
            model_cpkt_path (str): Path to the model checkpoint.
            model_ckpt (str): Model checkpoint filename.
            ema_model_ckpt (str): EMA model checkpoint filename.
        """
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1, save_dir="/content/drive/MyDrive/my_model"):
        """
        Saves the model's state dictionary.

        Args:
            run_name (str): Name of the run for creating sub-directory.
            epoch (int): Epoch number for checkpoint naming.
            save_dir (str): Directory to save the model checkpoint.
        """
        # Ensure the directory exists
        save_path = os.path.join(save_dir, run_name)
        os.makedirs(save_path, exist_ok=True)

        # Save the model state dict
        torch.save(self.model.state_dict(), os.path.join(save_path, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(save_path, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, f"optim.pt"))

        # print(f"Model saved to {checkpoint_path}")

    def prepare(self, args):
        """
        Prepares the data, optimizer, scheduler, loss function, and other components for training.

        Args:
            args: Arguments containing hyperparameters.
        """
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        print(f"train_dataloader type in prepare: {type(self.train_dataloader)}")
        print(f"val_dataloader type in prepare: {type(self.val_dataloader)}")
        print("check")
        batch = next(iter(self.train_dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch 'signal' shape: {batch['signal'].shape}")
        print(f"Batch 'label' shape: {batch['label'].shape if 'label' in batch else 'No label'}")
        print(f"Batch 'signal' type: {type(batch['signal'])}")
        print(f"Batch 'label' type: {type(batch['label']) if 'label' in batch else 'No label'}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        """
        Trains the model for the specified number of epochs.

        Args:
            args: Arguments containing hyperparameters.
        """
        pbar = progress_bar(range(args.epochs), total=args.epochs, leave=True) if is_notebook() else range(args.epochs)
        for epoch in pbar:

            logging.info(f"Starting epoch {epoch}:")
            _ = self.one_epoch(train=True)

            # Validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)

            # Log predictions
            if epoch % args.log_every_epoch == 0:
                self.log_signals()

        # Save model
        self.save_model(run_name=args.run_name, epoch=epoch)


def plot_3d_signals(signals):
    num_labels, num_dimensions, sequence_length, _ = signals.shape

    for label_idx in range(num_labels):
        fig, axes = plt.subplots(num_dimensions, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'Signals for label {label_idx + 1}')

        # Extract the 3D signal for the current label
        signal = signals[label_idx]  # Shape: [num_dimensions, sequence_length, 1]

        # Plot each dimension
        for dim_idx in range(num_dimensions):
            axes[dim_idx].plot(signal[dim_idx, :, 0], label=f'Dimension {dim_idx + 1}')
            axes[dim_idx].legend()
            axes[dim_idx].grid(True)
            axes[dim_idx].set_ylabel(f'Dim {dim_idx + 1}')

        axes[-1].set_xlabel('Sequence Length')

        plt.tight_layout()
        plt.show()


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
