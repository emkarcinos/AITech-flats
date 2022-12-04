import os
from abc import ABC
from pathlib import Path
from typing import Callable

import cv2 as cv
import numpy as np
import torch
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


def _load_data(input_dir: str, new_size: int | None = None):
    image_dir = Path(input_dir)
    categories_name = {}
    i = 0
    for file in os.listdir(image_dir):
        directory = os.path.join(image_dir, file)
        if os.path.isdir(directory):
            categories_name[i] = file
            i += 1

    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]

    train_img = []
    categories_count = len(folders)
    labels = []
    for directory in folders:
        count = 0
        for obj in directory.iterdir():
            if os.path.isfile(obj) and os.path.basename(os.path.normpath(obj)) != 'desktop.ini':
                try:
                    img = imread(obj)
                    if new_size is not None:
                        img = cv.resize(img, (new_size, new_size), interpolation=cv.INTER_AREA)
                    img = img / 255
                    train_img.append(img)
                    labels.append(os.path.basename(os.path.normpath(directory)))
                    count += 1
                except ValueError:
                    # This can happen when a file is broken, so let's omit it.
                    print(f'Broken file: {obj}')
    return {
        "values": np.array(train_img),
        "categories_count": categories_count,
        "labels": labels,
        "categories_name": categories_name
    }


class FlatsDataset(Dataset):
    def __init__(self, data, device):
        self.x = []
        for d in data['values']:
            self.x.append(transforms.ToTensor()(d).to(device))
        self.y = torch.LongTensor(LabelEncoder().fit_transform(data['labels'])).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class FlatsDatasetLoader(Dataset, ABC):
    def __init__(
            self,
            images_dir: str,
            resize_to: int or None = None,
            batch_size: int = 512,
            device: str = 'cpu',
            file_loader: Callable[[str, int or None], dict] = _load_data
    ):
        self.images_dir = images_dir
        self.resize_to = resize_to
        self.batch_size = batch_size
        self.device = device
        self.loader = file_loader
        self.train_loader = None
        self.test_loader = None
        self.classes_count = 0
        self.label_names = {}

    def load(self, verbose: bool = True):
        test_dir = os.path.join(self.images_dir, 'test')
        train_dir = os.path.join(self.images_dir, 'train')

        if verbose:
            print('Loading dataset from files...')
        test_raw = self.loader(test_dir, self.resize_to)
        train_raw = self.loader(train_dir, self.resize_to)

        self.classes_count = test_raw['categories_count']
        self.label_names = test_raw['categories_name']
        if verbose:
            print('Done. Creating PyTorch datasets...')
        train_set = FlatsDataset(train_raw, self.device)
        test_set = FlatsDataset(test_raw, self.device)

        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        if verbose:
            print('Done.')

    def get_train_loader(self) -> DataLoader:
        return self.train_loader

    def get_test_loader(self) -> DataLoader:
        return self.test_loader

    def get_label_names(self) -> dict:
        return self.label_names

    def get_classes_count(self) -> int:
        return self.classes_count
