import os
from abc import ABC
from pathlib import Path
from typing import Callable
import json

import cv2 as cv
import numpy as np
import torch
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms as transforms
import torchvision
import multiprocessing as mp


def _prepare_directory(input_dir: str):
    try:
        os.remove(os.path.join(input_dir, ".DS_Store"))
        print("DS_Store file removed")
    except FileNotFoundError:
        print("No DS_Store file found")


def _get_default_workers_count():
    return int(mp.cpu_count()/2)

class FlatsDataset(Dataset):
    def __init__(self, data, device):
        self.device = device
        self.x = []
        for d in data['values']:
            self.x.append(transforms.ToTensor()(d))
        self.y = torch.LongTensor(LabelEncoder().fit_transform(data['labels']))

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
    ):
        self.images_dir = images_dir
        self.resize_to = resize_to
        self.batch_size = batch_size
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.classes_count = 0
        self.label_names = {}
        _prepare_directory(self.images_dir)



    def load(self, train_set_ratio=0.8, workers_num=_get_default_workers_count(), verbose: bool=True, subset_size: int=None):
        transformation = transforms.Compose([
            transforms.Resize(self.resize_to),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f'Workers count: {_get_default_workers_count()}')
        
        if verbose:
            print('Loading dataset from files...')
        full_dataset = torchvision.datasets.ImageFolder(
            root=self.images_dir,
            transform=transformation
        )
        
        self.classes_count = len(full_dataset.classes)
        self.label_names = dict([(value, key) for key, value in full_dataset.class_to_idx.items()])

        if subset_size:
            full_dataset = torch.utils.data.Subset(full_dataset, np.random.choice(len(full_dataset), subset_size, replace=False))
    
        train_size = int(train_set_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size

        generator = torch.Generator()
        generator.manual_seed(0)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)


        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=workers_num,
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=workers_num,
            shuffle=False
        )

        print('Done.')

    def get_train_loader(self) -> DataLoader:
        return self.train_loader

    def get_test_loader(self) -> DataLoader:
        return self.test_loader

    def get_label_names(self) -> dict:
        return self.label_names

    def get_classes_count(self) -> int:
        return self.classes_count
    
    def __str__(self):
        json_obj = {
            "Full dataset count: ": len(self.train_loader.dataset) + len(self.test_loader.dataset),
            "Train dataset count: ": len(self.train_loader.dataset),
            "Test dataset count: ": len(self.test_loader.dataset),
            "Classes count: ": self.classes_count,
            "Label names: ": self.get_label_names()
        }
        json_formatted_str = json.dumps(json_obj, indent=2)
        return json_formatted_str

