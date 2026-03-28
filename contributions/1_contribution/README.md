
# A. Feature-Aware Training Curriculum

---

## Overview

Burgert et al. introduce a controlled suppression framework that *measures* CNN feature reliance across shape, texture, and color cues. Their paper is a measurement contribution — it does not propose any mechanism to change what a model relies on.

This contribution reframes that as a control problem:

> **If feature reliance is measurable, can it be actively reduced through training?**

We propose a **Feature-Aware Training Curriculum** that dynamically mixes clean and feature-suppressed images during training, using the same suppression operations the paper uses for evaluation. The curriculum progressively increases suppression pressure as training advances, forcing the model to learn representations that do not depend on any single feature cue.

Evaluated on CIFAR-10 with ResNet-18 and assessed using the paper's own reliance framework:

| Feature       | Standard reliance | Curriculum reliance | Δ            |
|---------------|------------------|---------------------|--------------|
| Local shape   | 0.6191           | 0.4064              | ↓ 0.2127     |
| Global shape  | 0.5236           | 0.4147              | ↓ 0.1089     |
| Texture       | 0.2601           | 0.0503              | ↓ 0.2098     |
| Color         | 0.0944           | 0.0394              | ↓ 0.0550     |

Clean accuracy cost: **−2.02pp** (84.84% → 82.82%).

---

## Repository Structure

```

contribution/
├── suppression_utils.py       # Tensor-native suppression operations
│                              # (patch shuffle, bilateral filter, grayscale)
├── curriculum_dataset.py      # CurriculumDataset with dynamic schedule
├── train.py                   # Training loop for standard and curriculum modes
├── gradcam_viz.py             # Grad-CAM visualization (standard vs curriculum)
├── logs/
│   └── reliance_comparison.json   # Full evaluation results
└── models/
├── standard.pt            # Trained standard ResNet-18 checkpoint
└── curriculum.pt          # Trained curriculum ResNet-18 checkpoint

data/
└── cifar-10-batches-py/       # CIFAR-10 dataset (auto-downloaded if absent)

feature-aware-training-curriculum.ipynb   # Full Kaggle notebook (end-to-end)

```

---

## How It Works

### Suppression Operations

Three suppression modes are implemented in `suppression_utils.py`, adapted from the paper's `transforms.py` as tensor-native operations compatible with PyTorch's CIFAR-10 pipeline:

| Mode      | Operation                               | What it destroys            |
|-----------|------------------------------------------|-----------------------------|
| `shape`   | Patch shuffle (`grid_size = 6`)           | Local spatial structure     |
| `texture` | Bilateral filter (`d = 11`, `σ_c = 170`)  | Surface texture detail      |
| `color`   | Grayscale conversion                     | Color information           |

> **Note on parameters:** Bilateral filtering uses stronger parameters than the paper's default (`d = 5`, `σ_c = 75`) to compensate for CIFAR-10's 32×32 resolution, where weaker smoothing has negligible visible effect.

---

### Dynamic Curriculum Schedule

At the start of each epoch, the mixing probabilities update according to:

```

p_normal(t) = 0.70 - 0.60 × (t - 1) / (T - 1)   # 0.70 → 0.10
p_shape     = p_texture = p_color = (1 - p_normal) / 3

````

| Epoch | Normal | Shape | Texture | Color |
|-------|--------|--------|----------|--------|
| 1     | 70%    | 10%    | 10%      | 10%    |
| 15    | 40%    | 20%    | 20%      | 20%    |
| 30    | 10%    | 30%    | 30%      | 30%    |

The model starts on mostly clean images to learn the basic task, then is progressively forced to operate without individual feature cues. Because it never knows which version it receives, it cannot specialize and must learn feature-robust representations.

---

## Installation

```bash
# GPU (CUDA 11.8) — matches Kaggle T4/P100 environment
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install grad-cam opencv-python tqdm
````

CIFAR-10 downloads automatically on first run. No manual data preparation is required.

---

## Usage

### Option 1 — Kaggle Notebook (Recommended)

Open `feature-aware-training-curriculum.ipynb` in Kaggle with GPU accelerator enabled. Run all cells in order. The notebook covers installation, training both models, evaluation, and result logging end-to-end.

---

### Option 2 — Script

```bash
# Train standard model
python contribution/train.py --mode standard --epochs 30 --batch_size 128

# Train curriculum model
python contribution/train.py --mode curriculum --epochs 30 --batch_size 128
```

Checkpoints are saved to `contribution/models/` by default.

---

### Evaluate Feature Reliance

Run the evaluation cell in the notebook, or use the script below:

```python
import sys
sys.path.append('contribution')

from suppression_utils import apply_suppression
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

device = torch.device('cuda')

def load_model(path):
    m = models.resnet18(weights=None, num_classes=10).to(device)
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m

std_model = load_model('contribution/models/standard.pt')
cur_model = load_model('contribution/models/curriculum.pt')

# Results print to console and save to contribution/logs/reliance_comparison.json
```

---

## Results

Full results are saved in:

```
contribution/logs/reliance_comparison.json
```

Example output:

```json
{
  "accuracy": {
    "original":     { "standard": 0.8484, "curriculum": 0.8282 },
    "local_shape":  { "standard": 0.2293, "curriculum": 0.4218 },
    "global_shape": { "standard": 0.3248, "curriculum": 0.4135 },
    "texture":      { "standard": 0.5883, "curriculum": 0.7779 },
    "color":        { "standard": 0.7540, "curriculum": 0.7888 }
  },
  "reliance": {
    "local_shape":  { "standard": 0.6191, "curriculum": 0.4064 },
    "global_shape": { "standard": 0.5236, "curriculum": 0.4147 },
    "texture":      { "standard": 0.2601, "curriculum": 0.0503 },
    "color":        { "standard": 0.0944, "curriculum": 0.0394 }
  }
}
```

---

## Design Decisions and Scope

| Decision           | What we did       | Why                                                                   |
| ------------------ | ----------------- | --------------------------------------------------------------------- |
| Architecture       | ResNet-18         | Compute constraints (Kaggle T4); protocol is architecture-agnostic    |
| Dataset            | CIFAR-10          | Accessible, well-benchmarked; allows fast iteration                   |
| Suppression ops    | One per feature   | Follows primary transform from each pair in Table 1 of Burgert et al. |
| Bilateral params   | d = 11, σ_c = 170 | Stronger suppression needed at 32×32 resolution                       |
| Schedule endpoints | 0.70 → 0.10       | Ensures sufficient clean data early; strong suppression late          |

The standard model's local shape reliance (0.6191, relative accuracy ≈ 0.27) matches the paper's ResNet-50 result on ImageNet (0.276), validating that our baseline behaves as expected despite the architecture and dataset difference.

---

## Relation to Base Paper

|         | Burgert et al.           | This contribution               |
| ------- | ------------------------ | ------------------------------- |
| Goal    | Measure feature reliance | Control feature reliance        |
| Method  | Suppression at test time | Suppression during training     |
| Signal  | Accuracy drop            | Accuracy drop (same metric)     |
| Finding | CNNs rely on local shape | Curriculum reduces all reliance |

The evaluation uses the paper's own measurement framework applied to both models, ensuring direct comparability.

---

## Citation

If you use this curriculum extension, please also cite the base paper:

```bibtex
@misc{burgert2025featurereliance,
  title        = {ImageNet-trained CNNs are not biased towards texture:
                  Revisiting feature reliance through controlled suppression},
  author       = {Tom Burgert and Oliver Stoll and Paolo Rota and Begüm Demir},
  year         = {2025},
  eprint       = {2509.20234},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV}
}
```

`
