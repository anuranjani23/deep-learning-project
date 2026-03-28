
import random
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/kaggle/working/contribution')
from suppression_utils import apply_suppression


def get_curriculum_probs(epoch: int, total_epochs: int) -> dict:
    """
    Dynamic schedule: starts easy (mostly normal images),
    ramps up suppression difficulty as training progresses.

    Epoch 1  → normal=0.70, shape=0.10, texture=0.10, color=0.10
    Epoch T  → normal=0.10, shape=0.30, texture=0.30, color=0.30
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)  # 0.0 → 1.0
    p_normal  = 0.70 - 0.60 * progress          # 0.70 → 0.10
    p_shared  = (1.0 - p_normal) / 3            # equal split for the rest
    return {
        'normal' : round(p_normal,  4),
        'shape'  : round(p_shared,  4),
        'texture': round(p_shared,  4),
        'color'  : round(p_shared,  4),
    }


class CurriculumDataset(Dataset):
    def __init__(self, base_dataset, grid_size=6, normalize=None):
        self.dataset   = base_dataset
        self.modes     = ['normal', 'shape', 'texture', 'color']
        self.weights   = [0.70, 0.10, 0.10, 0.10]   # epoch 1 defaults
        self.grid_size = grid_size
        self.normalize = normalize

    def set_epoch(self, epoch: int, total_epochs: int):
        """Call at the start of each epoch to update the mixing schedule."""
        probs = get_curriculum_probs(epoch, total_epochs)
        self.weights = [
            probs['normal'],
            probs['shape'],
            probs['texture'],
            probs['color'],
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        mode = random.choices(self.modes, weights=self.weights)[0]
        img  = apply_suppression(img, mode, self.grid_size)
        if self.normalize:
            img = self.normalize(img)
        return img, label, mode
