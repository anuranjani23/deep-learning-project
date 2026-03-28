# Feature Reliance: Measurement, Control, and Cross-Domain Analysis

**Extension of:** *"ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression"* — Tom Burgert, Oliver Stoll, Paolo Rota, Begüm Demir @ NeurIPS 2025 (Oral)

[![arXiv](https://img.shields.io/badge/arXiv-2509.20234-b31b1b.svg)](https://arxiv.org/abs/2509.20234)
[![Original Repo](https://img.shields.io/badge/Original%20Repo-tomburgert%2Ffeature--reliance-blue)](https://github.com/tomburgert/feature-reliance)

**Course:** CSL4020 Deep Learning — Project (2026) 

---

## What this repo contains

This repository builds on the official implementation of Burgert et al. and extends it in two directions as part of a course project. The base paper proposes a controlled suppression framework to *measure* CNN feature reliance — it does not propose any mechanism to change or control that reliance. We address this gap along two independent axes:

| Contribution | What |
|---|---|
| **A. Feature-Aware Training Curriculum** | Training-time intervention that actively reduces feature reliance using a dynamic suppression schedule |
| **B. Cross-Domain CLIP Similarity & Robust Fine-tuning** | Extends the measurement framework to CLIP embedding space and cross-domain robust training across CV, medical, and remote sensing domains |

Both contributions use the paper's suppression operations (patch shuffle, bilateral filter, grayscale) as their foundation and evaluate using the paper's reliance measurement framework.

## Replication & Execution

More information on running the code and replicating results for each contribution is provided in the specific **README** files located within their respective directories:

* **Contribution 1:** `contributions/1_contribution/README.md`
* **Contribution 2:** `contributions/2_contribution/README.md`

Each folder-level README contains the necessary environment specifications, dependency requirements, and command-line instructions unique to that portion of the project.

## Key results

### Contribution A — Feature-Aware Training Curriculum (CIFAR-10, ResNet-18)

| Feature | Standard reliance | Curriculum reliance | Δ |
|---|---|---|---|
| Local shape | 0.6191 | 0.4064 | **↓ 0.2127** |
| Global shape | 0.5236 | 0.4147 | **↓ 0.1089** |
| Texture | 0.2601 | 0.0503 | **↓ 0.2098** |
| Color | 0.0944 | 0.0394 | **↓ 0.0550** |

Clean accuracy cost: **−2.02pp** (84.84% → 82.82%).

### Contribution B — Cross-Domain Robust Training

**CLIP prompt engineering on MS-COCO** (200 samples, pretrained ViT-B/32):

| Label mode | clip_basic | shape | texture | color | base_caption |
|---|---|---|---|---|---|
| caption | 0.3018 | 0.2850 | 0.2881 | 0.2919 | 0.3037 |
| noun | 0.2107 | 0.1780 | 0.1748 | 0.1851 | 0.3037 |
| short | 0.2607 | 0.2316 | 0.2303 | 0.2389 | 0.3037 |

**Best robust models per domain** (test accuracy / AP under all suppression conditions):

| Domain | Dataset | Best training strategy | Baseline acc | Shape-removed | Texture-removed | Color-removed |
|---|---|---|---|---|---|---|
| Computer Vision | STL10 | Shape-removed training | 0.9760 | 0.9354 | 0.9531 | 0.9538 |
| Medical | BloodMNIST | Shape + color augmentation | 0.9872 | 0.9705 | 0.9230 | 0.9790 |
| Remote Sensing | DeepGlobe | Texture + color augmentation | 0.8539 | 0.8320 | 0.8483 | 0.8413 |

---

## Data preparation

### Contribution A
CIFAR-10 downloads automatically via `torchvision.datasets`. No setup needed.

### Contribution B + base paper

**Computer Vision:** Caltech101, Flowers102, OxfordPet, STL10 download automatically via `torchvision.datasets`. ImageNet requires the official ILSVRC 2012 split.

**Medical:** BloodMNIST, RetinaMNIST and other MedMNIST datasets are handled via the `medmnist` package included in `requirements.txt`.

**Remote Sensing:**
- DeepGlobe: [Kaggle link] — after download, run `scripts/preprocess_deepglobe.py`
- RSD46-WHU: [Hugging Face link] — after download, run `scripts/preprocess_rsd46whu.py`
- PatternNet: [Download here]
- UCMerced: [Download here]
- AID: [Hugging Face link]

---

## How the contributions relate to the base paper

| | Burgert et al. | Contribution A | Contribution B |
|---|---|---|---|
| Goal | Measure feature reliance | Control feature reliance | Generalize measurement + improve robustness |
| Method | Suppression at test time | Suppression during training | CLIP similarity + domain-specific fine-tuning |
| Signal | Accuracy drop | Accuracy drop (same metric) | CLIP cosine similarity + cross-domain accuracy |
| Dataset | ImageNet, multi-domain | CIFAR-10 | MS-COCO, STL10, BloodMNIST, DeepGlobe |
| Model | ResNet-50 | ResNet-18 | CLIP ViT-B/32, ResNet-50 |
| Finding | CNNs rely on local shape | Curriculum reduces all reliance types | Domain bias confirmed; robust training strategies identified per domain |

**The base paper establishes that feature reliance is measurable. Contribution A shows it is controllable through training. Contribution B shows it generalizes across modalities and can be targeted through domain-specific augmentation.**

---

## Design decisions (Contribution A)

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | ResNet-18 | Compute constraints (Kaggle T4); protocol is architecture-agnostic |
| Dataset | CIFAR-10 | Well-benchmarked; enables fast iteration |
| Suppression ops | One per feature type | Follows primary transform from Table 1 of Burgert et al. |
| Bilateral params | d=11, σ_c=170 | Stronger suppression required at 32×32 resolution |
| Schedule endpoints | 0.70 → 0.10 | Sufficient clean data early; maximum suppression pressure late |

The standard model's local shape reliance (0.6191, relative accuracy ≈ 0.27) matches the paper's ResNet-50 result on ImageNet (0.276), validating the baseline.

---

## Citation

If you use this code, please cite the base paper:

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

```bibtex
@InProceedings{DeepGlobe18,
  author    = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan
               and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
  title     = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
  booktitle = {CVPR Workshops},
  year      = {2018}
}
```
