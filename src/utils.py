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

    # Fixed sequence length
    fixed_sequence_length = 4*1024

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

        # Truncate or pad sequences to the fixed length of 4*1024
        acce_data_x = acce_data_x[:fixed_sequence_length] if len(acce_data_x) > fixed_sequence_length else np.pad(
            acce_data_x, (0, fixed_sequence_length - len(acce_data_x)), 'constant')
        acce_data_y = acce_data_y[:fixed_sequence_length] if len(acce_data_y) > fixed_sequence_length else np.pad(
            acce_data_y, (0, fixed_sequence_length - len(acce_data_y)), 'constant')
        acce_data_z = acce_data_z[:fixed_sequence_length] if len(acce_data_z) > fixed_sequence_length else np.pad(
            acce_data_z, (0, fixed_sequence_length - len(acce_data_z)), 'constant')

        # Stack the accelerometer data to form a sample with shape (num_channels, sequence_length)
        sample_data = np.stack([acce_data_x, acce_data_y, acce_data_z], axis=0)  # Shape: (3, 256)

        # Append the sample data to the signals list
        signals.append(sample_data)

        # Append the corresponding label to the labels list
        labels.append(row['activity'])

    # Convert the signals list to a numpy array with shape (num_samples, num_channels, fixed_sequence_length)
    signals_array = np.array(signals)
    print(signals_array)
    print(signals_array.shape)
    signals_array = np.expand_dims(signals_array, axis=-1)
    print(signals_array)
    print(signals_array.shape)
    # Convert the labels list to a numpy array with shape (num_samples,)
    labels_array = np.array(labels)
    labels_array = np.array([label_mapping[label] for label in labels_array])
    print(labels_array)
    print(labels_array.shape)
    return signals_array, labels_array


def get_data(args):
    # train_transforms = torchvision.transforms.Compose([
    #     T.Resize(args.img_size + int(.25*args.img_size)),  # args.img_size + 1/4 *args.img_size
    #     T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
    #     T.ToTensor(),
    #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    #
    # val_transforms = torchvision.transforms.Compose([
    #     T.Resize(args.img_size),
    #     T.ToTensor(),
    #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    # train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.train_folder), transform=train_transforms)
    # val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.val_folder), transform=val_transforms)
    # if args.slice_size>1:
    #     train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
    #     val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), args.slice_size))
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_dataset = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
    # Instantiate the dataset

    signals, labels = extract_signal_data()

    # Normalize each axis of the 3D signals independently (axis-wise normalization)
    num_samples, num_channels, sequence_length = signals.shape
    scaler = StandardScaler()

    # Apply normalization to each channel (x, y, z) independently
    normalized_signals = signals.copy()
    for i in range(num_channels):  # Iterate over channels (x, y, z)
        normalized_signals[:, i, :] = scaler.fit_transform(signals[:, i, :])

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