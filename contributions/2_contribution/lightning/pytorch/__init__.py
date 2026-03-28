"""
Local shim for `lightning.pytorch` that re-exports pytorch_lightning symbols.
This avoids requiring the external `lightning` package.
"""

from pytorch_lightning import *  # noqa: F401,F403

# Explicitly re-export common classes for clarity
from pytorch_lightning import LightningDataModule, LightningModule, Trainer  # noqa: F401
