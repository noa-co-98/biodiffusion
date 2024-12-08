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

cifar_labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")
alphabet_labels = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")
BASE_DIR =  Path('/content/data_publish_v2') / 'data_publish_v2'
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

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def untar_data(url, force_download=False, base='./datasets'):
    d = FastDownload(base=base)
    return d.get(url, force=force_download, extract_key='data')


def get_alphabet(args):
    get_kaggle_dataset("alphabet", "thomasqazwsxedc/alphabet-characters-fonts-dataset")
    train_transforms = T.Compose([
        T.Grayscale(),
        T.ToTensor(),])
    train_dataset = torchvision.datasets.ImageFolder(root="./alphabet/Images/Images/", transform=train_transforms)
    if args.slice_size>1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return train_dataloader, None

def get_cifar(cifar100=False, img_size=64):
    "Download and extract CIFAR"
    cifar10_url = 'https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz'
    cifar100_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz'
    if img_size==32:
        return untar_data(cifar100_url if cifar100 else cifar10_url)
    else:
        get_kaggle_dataset("datasets/cifar10_64", "joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution")
        return Path("datasets/cifar10_64/cifar10-64")

def get_kaggle_dataset(dataset_path, # Local path to download dataset to
                dataset_slug, # Dataset slug (ie "zillow/zecon")
                unzip=True, # Should it unzip after downloading?
                force=False # Should it overwrite or error if dataset_path exists?
               ):
    '''Downloads an existing dataset and metadata from kaggle'''
    if not force and Path(dataset_path).exists():
        return Path(dataset_path)
    api.dataset_metadata(dataset_slug, str(dataset_path))
    api.dataset_download_files(dataset_slug, str(dataset_path))
    if unzip:
        zipped_file = Path(dataset_path)/f"{dataset_slug.split('/')[-1]}.zip"
        import zipfile
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            zip_ref.extractall(Path(dataset_path))
        zipped_file.unlink()

def one_batch(dl):
    batch = next(iter(dl))
    signals, labels = batch['signal'], batch['label']
    print("type check")
    print(type(signals))  # Should be <class 'torch.Tensor'>
    print(type(labels))   # Should be <class 'torch.Tensor'>
    return signals, labels

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


class MultiChannelSignalDataset(Dataset):
    def __init__(self, signals, labels=None, transform=None):
        """
        Args:
            signals (np.array): Numpy array of shape (num_samples, num_channels, sequence_length).
            labels (np.array, optional): Numpy array of shape (num_samples,) containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx] if self.labels is not None else None

        sample = {'signal': torch.tensor(signal, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long) if label is not None else None}

        if self.transform:
            sample = self.transform(sample)

        return sample

def extract_signal_data():
    all_files_df = pd.DataFrame({'path': list(BASE_DIR.glob('*/*.txt'))})
    print(all_files_df.shape)
    print(all_files_df.head())
    all_files_df['exp_code'] = all_files_df['path'].map(lambda x: x.parent.stem)
    all_files_df['activity'] = all_files_df['exp_code'].map(lambda x: '_'.join(x.split('_')[1:]))
    all_files_df['person'] = all_files_df['exp_code'].map(lambda x: x.split('_')[0])
    all_files_df['data_src'] = all_files_df['path'].map(lambda x: x.stem)

    data_df = all_files_df.pivot_table(values='path',
                                       columns='data_src',
                                       index=['activity', 'person'],
                                       aggfunc='first'). \
        reset_index(). \
        dropna(axis=1)  # remove mostly empty columns
      # Sliding window parameters
    window_size = 400  # 2 seconds of data at 200 Hz
    step_size = window_size // 2  # 50% overlap
    start_limit = 1500  # Minimum starting index for the sliding window

    # Lists to store the accelerometer signal data and labels for all samples
    signals = []
    labels = []

    # Iterate over rows in the data_df
    for index, row in data_df.iterrows():
        # Read accelerometer data
        acce_df = pd.read_csv(row['acce'], sep=" ", header=None, names=['x', 'y', 'z'], skiprows=1)

        # Extract accelerometer data for x, y, z axes
        acce_data_x = acce_df['x'].values  # Acceleration 'x' data
        acce_data_y = acce_df['y'].values  # Acceleration 'y' data
        acce_data_z = acce_df['z'].values  # Acceleration 'z' data

        # Stack accelerometer data into a single array (shape: (num_samples, num_channels))
        acce_data = np.stack([acce_data_x, acce_data_y, acce_data_z], axis=1)  # Shape: (num_samples, 3)

        # Ensure the signal starts from at least index 1500
        acce_data = acce_data[start_limit:]  # Shape: (remaining_samples, 3)
        num_samples = len(acce_data)

        # Apply sliding window to the accelerometer data
        for start_idx in range(0, num_samples - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = acce_data[start_idx:end_idx]  # Shape: (window_size, 3)

            # Transpose to match (num_channels, sequence_length)
            window = window.T  # Shape: (3, window_size)
            
            # Append the window data to the signals list
            signals.append(window)

            # Append the corresponding label to the labels list
            labels.append(row['activity'])

    # Convert the signals list to a numpy array with shape (num_samples, num_channels, window_size)
    signals_array = np.array(signals)
    print(signals_array)
    print(signals_array.shape)
    signals_array = np.expand_dims(signals_array, axis=-1)  # Add a channel dimension for models expecting 4D input
    print(signals_array)
    print(signals_array.shape)

    # Convert the labels list to a numpy array with shape (num_samples,)
    labels_array = np.array(labels)
    labels_array = np.array([label_mapping[label] for label in labels_array])
    print(labels_array)
    print(labels_array.shape)

    return signals_array, labels_array

def get_data(args):

    signals, labels = extract_signal_data()

    # Normalize each axis of the 3D signals independently (axis-wise normalization)
    # Assuming signals have the shape (num_samples, num_channels, sequence_length, 1)
    num_samples, num_channels, sequence_length, _ = signals.shape
    
    means = np.zeros(num_channels)
    stds = np.zeros(num_channels)
    # Normalize each axis (channel) independently
    for i in range(num_channels):
        axis_data = signals[:, i, :, 0]  # Extract data for channel i (shape: num_samples, sequence_length)
        # mean = np.nanmean(axis_data, axis=1, keepdims=True)  # Mean per sample for the current channel
        # std = np.nanstd(axis_data, axis=1, keepdims=True)    # Std per sample for the current channel
        mean = np.nanmean(axis_data)  # Mean for the current channel
        std = np.nanstd(axis_data)    # Std for the current channel
        signals[:, i, :, 0] = (axis_data - mean) / (std + 1e-8)  # Normalize and avoid division by zero
    # # Normalize each axis (channel) independently
    # for i in range(num_channels):
    #     axis_data = signals[:, i, :, 0]  # Extract data for channel i (shape: num_samples, sequence_length)
    #     mean = np.nanmean(axis_data, axis=1, keepdims=True)  # Mean per sample for the current channel
    #     std = np.nanstd(axis_data, axis=1, keepdims=True)    # Std per sample for the current channel
    #     signals[:, i, :, 0] = (axis_data - mean) / (std + 1e-8)  # Normalize and avoid division by zero



    signal_dataset = MultiChannelSignalDataset(signals=signals, labels=labels)
    #print(signal_dataset)
    #print("check 1")
    #print(signal_dataset.__getitem__)
    train_size = int(0.8 * len(signal_dataset))
    val_size = len(signal_dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(signal_dataset, [train_size, val_size])
    # Create a DataLoader
    print(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=4)
    print(train_dataloader)
    return train_dataloader, val_dataloader


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
