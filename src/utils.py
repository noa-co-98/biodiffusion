import os, random
from pathlib import Path
from kaggle import api
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from fastdownload import FastDownload
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

label_mapping = {
    'bag1': 0, 'bag2': 0, 'bag_low2': 0, 'bag_low3': 0, 'bag_normal1': 0,
    'bag_normal2': 0, 'bag_side1': 0, 'bag_speed1': 0, 'bag_speed2': 0,
    'bag_stop1': 0, 'bag_test1': 0, 'body1': 1, 'body2': 1, 'body3': 1,
    'body_backward1': 1, 'body_backward2': 1, 'body_backward3': 1,
    'body_backward4': 1, 'body_fast1': 1, 'body_normal1': 1, 'body_side1': 1,
    'body_slow1': 1, 'body_stop1': 1, 'body_test1': 1, 'handheld1': 2,
    'handheld2': 2, 'handheld3': 2, 'handheld_normal1': 2, 'handheld_side3': 2,
    'handheld_side4': 2, 'handheld_side_test2': 2, 'handheld_speed1': 2,
    'handheld_speed2': 2, 'handheld_test1': 2, 'leg1': 3, 'leg2': 3,
    'leg_front1': 3, 'leg_front2': 3, 'leg_front3': 3, 'leg_new1': 3,
    'leg_new2': 3, 'lopata1': 4
}

class MultiChannelSignalDataset(Dataset):
    def __init__(self, signals, labels=None, transform=None):
        """
        Dataset for multi-channel sensor signals.
        Args:
            signals (np.array): Array of shape (num_samples, num_channels, seq_length, 1).
            labels (np.array, optional): Array of labels for each sample.
            transform (callable, optional): Optional transform to apply.
        """
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx] if self.labels is not None else None
        sample = {'signal': torch.tensor(signal, dtype=torch.float32),
                  'label': torch.tensor(label, dtype=torch.long) if label is not None else None}
        if self.transform:
            sample = self.transform(sample)
        return sample

def extract_signal_data():
    BASE_DIR = Path('/content/data_publish_v2') / 'data_publish_v2'
    all_files_df = pd.DataFrame({'path': list(BASE_DIR.glob('*/*.txt'))})
    all_files_df['exp_code'] = all_files_df['path'].map(lambda x: x.parent.stem)
    all_files_df['activity'] = all_files_df['exp_code'].map(lambda x: '_'.join(x.split('_')[1:]))
    all_files_df['person'] = all_files_df['exp_code'].map(lambda x: x.split('_')[0])
    all_files_df['data_src'] = all_files_df['path'].map(lambda x: x.stem)

    data_df = all_files_df.pivot_table(values='path',
                                       columns='data_src',
                                       index=['activity', 'person'],
                                       aggfunc='first').reset_index().dropna(axis=1)

    # Sliding window parameters
    window_size = 400      # e.g., 2 seconds at 200 Hz
    step_size = window_size // 2  # 50% overlap
    start_limit = 1500     # Minimum starting index

    signals = []
    labels = []

    for index, row in data_df.iterrows():
        acce_df = pd.read_csv(row['acce'], sep=" ", header=None,
                              names=['x', 'y', 'z'], skiprows=1)
        acce_data = np.stack([acce_df['x'].values, acce_df['y'].values, acce_df['z'].values], axis=1)
        acce_data = acce_data[start_limit:]
        num_samples = len(acce_data)
        for start_idx in range(0, num_samples - window_size + 1, step_size):
            window = acce_data[start_idx:start_idx+window_size].T  # Shape: (3, window_size)
            signals.append(window)
            labels.append(row['activity'])

    signals_array = np.array(signals)
    # Expand dims to add a spatial dimension (height=1) for 2D convolutions
    signals_array = np.expand_dims(signals_array, axis=-1)
    labels_array = np.array([label_mapping[label] for label in labels])
    return signals_array, labels_array

def get_data(args):
    signals, labels = extract_signal_data()
    # Normalize each channel independently
    num_samples, num_channels, sequence_length, _ = signals.shape
    # Lists to store the per-channel means and stds
    channel_means = []
    channel_stds = []

    for i in range(num_channels):
        # Extract data for channel i (shape: [num_samples, sequence_length])
        axis_data = signals[:, i, :, 0]
        mean = np.nanmean(axis_data)
        std = np.nanstd(axis_data)
        channel_means.append(mean)
        channel_stds.append(std)
        
        # Normalize each channel: (data - mean) / (std + epsilon)
        signals[:, i, :, 0] = (axis_data - mean) / (std + 1e-8)

    # Save channel_means and channel_stds (e.g., as part of your model or config)
    print("Channel Means:", channel_means)
    print("Channel Stds:", channel_stds)

    signal_dataset = MultiChannelSignalDataset(signals=signals, labels=labels)
    train_size = int(0.8 * len(signal_dataset))
    val_size = len(signal_dataset) - train_size
    train_dataset, val_dataset = random_split(signal_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=2 * args.batch_size, shuffle=False, num_workers=4)
    return train_dataloader, val_dataloader

def set_seed(s, reproducible=False):
    """Set random seed for reproducibility."""
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s % (2**32-1))
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
