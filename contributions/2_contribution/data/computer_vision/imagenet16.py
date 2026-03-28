from typing import Optional
import os
import glob

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from lightning.pytorch import LightningDataModule

from PIL import Image


class ImageNet16Dataset(Dataset):
    def __init__(self, dataset_path: str, split: str = None, transform: Optional[transforms.Compose] = None):
        self.dataset_path = self._resolve_root(dataset_path)
        self.split = split
        self.transform = transform

        split_dir = self._split_dir(self.split)
        self.image_dir = os.path.join(self.dataset_path, split_dir)
        if split_dir == 'test' and not os.path.isdir(self.image_dir):
            fallback_dir = os.path.join(self.dataset_path, 'val')
            if os.path.isdir(fallback_dir):
                self.image_dir = fallback_dir
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f'ImageNet16 split directory not found: {self.image_dir}. '
                f'Expected train/val folders inside {self.dataset_path}.'
            )

        self.classes = self._load_class_order(self.dataset_path, self.image_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.targets = []
        self._read_image_paths()

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        target = self.targets[idx]

        image = Image.open(image_path)
        target = torch.tensor(target)

        # some images (e.g. class car side) are grayscale, convert them to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)

    def _read_image_paths(self):
        for cls_name in self.classes:
            cls_folder = os.path.join(self.image_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for ext in ('*.JPEG', '*.jpg', '*.jpeg', '*.png'):
                for img_path in sorted(glob.glob(os.path.join(cls_folder, ext))):
                    self.image_paths.append(img_path)
                    self.targets.append(self.class_to_idx[cls_name])

        self.image_paths = np.array(self.image_paths)
        self.targets = np.array(self.targets)

    @staticmethod
    def _split_dir(split: Optional[str]) -> str:
        if split is None or split == 'train':
            return 'train'
        if split in ('validation', 'val'):
            return 'val'
        if split == 'test':
            return 'test'
        return split

    @staticmethod
    def _resolve_root(dataset_path: str) -> str:
        if os.path.isdir(os.path.join(dataset_path, 'train')) or os.path.isdir(os.path.join(dataset_path, 'val')):
            return dataset_path
        candidate = os.path.join(dataset_path, 'imagenet16')
        if os.path.isdir(os.path.join(candidate, 'train')) or os.path.isdir(os.path.join(candidate, 'val')):
            return candidate
        return dataset_path

    @staticmethod
    def _load_class_order(dataset_root: str, image_dir: str):
        map_path = os.path.join(dataset_root, 'map.txt')
        if not os.path.isfile(map_path):
            return sorted([d.name for d in os.scandir(image_dir) if d.is_dir()])

        wnids = []
        with open(map_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                cleaned = line.replace('\xa0', ' ').strip()
                if not cleaned or '=' not in cleaned:
                    continue
                _, right = cleaned.split('=', 1)
                wnid = right.strip()
                if wnid:
                    wnids.append(wnid)

        available = {d.name for d in os.scandir(image_dir) if d.is_dir()}
        ordered = [wnid for wnid in wnids if wnid in available]
        if len(ordered) == len(available):
            return ordered
        # Fallback to sorted directory names if map is incomplete.
        return sorted(available)


class ImageNet16DataModule(LightningDataModule):
    def __init__(
        self,
        root_path: str,
        batch_size: int,
        num_workers: int,
        train_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.dataset_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageNet16Dataset(
                dataset_path=self.dataset_path,
                split='train',
                transform=self.train_transform,
                
            )
            self.val_dataset = ImageNet16Dataset(
                dataset_path=self.dataset_path,
                split='validation',
                transform=self.test_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = ImageNet16Dataset(
                dataset_path=self.dataset_path,
                split='test',
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
