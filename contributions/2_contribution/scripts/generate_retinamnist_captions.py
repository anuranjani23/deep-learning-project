#!/usr/bin/env python3
"""
Generate captions for RetinaMNIST images using an image captioning model.

Default backend uses BLIP (transformers). This is optional and will download
model weights on first run.

Example:
  python3 scripts/generate_retinamnist_captions.py \
    --root ./dataset/tomburgert \
    --split train \
    --backend blip \
    --out ./dataset/retinamnist_captions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional
import re
import numpy as np

from PIL import Image

from medmnist import RetinaMNIST


def iter_indices(total: int, max_samples: Optional[int]) -> Iterable[int]:
    if max_samples is None or max_samples >= total:
        return range(total)
    return range(max_samples)


def load_dataset(root: str, split: str) -> RetinaMNIST:
    return RetinaMNIST(root=root, split=split, download=True, size=224)


def init_blip(device: str):
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "BLIP backend requires transformers + torch. Install with `pip install transformers`."
        ) from exc

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # Use safetensors to avoid torch.load restrictions on older torch versions.
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True,
    ).to(device)
    model.eval()
    return processor, model, torch


def init_blip2(device: str):
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "BLIP-2 backend requires transformers + torch. Install with `pip install transformers`."
        ) from exc

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    return processor, model, torch


def caption_with_blip(images: List[Image.Image], device: str, processor, model, torch_mod) -> List[str]:
    with torch_mod.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=30)
        captions = processor.batch_decode(out, skip_special_tokens=True)
    return captions


def caption_with_blip2(images: List[Image.Image], device: str, processor, model, torch_mod) -> List[str]:
    with torch_mod.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device, torch_mod.float16)
        out = model.generate(**inputs, max_new_tokens=40)
        captions = processor.batch_decode(out, skip_special_tokens=True)
    return captions


def caption_dummy(images: List[Image.Image]) -> List[str]:
    # Simple fallback captions for testing the pipeline.
    return ["a retinal fundus image"] * len(images)


SHAPE_WORDS = {
    "round", "circular", "oval", "square", "rectangular", "triangular", "long", "short",
    "curved", "straight", "thin", "thick", "wide", "narrow", "branching", "elongated",
}
TEXTURE_WORDS = {
    "smooth", "rough", "furry", "hairy", "striped", "spotted", "speckled", "grainy", "matte",
    "glossy", "shiny", "wrinkled", "bumpy", "soft", "hard", "velvety", "patchy",
}
COLOR_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "gold", "silver", "beige", "tan", "maroon", "cyan", "magenta",
}

NEUTRAL_TEMPLATES = [
    "a retinal fundus image",
    "a retinal fundus photograph",
    "an ophthalmic fundus image",
    "a medical retinal screening image",
    "a retinal imaging sample",
]

GENERIC_TEMPLATES = [
    "a photo of a retina",
    "a retinal image",
    "a fundus photo",
    "a medical image of the retina",
    "a close-up retinal photograph",
]

RETINAMNIST_DISEASE_NAMES = {
    0: "no diabetic retinopathy",
    1: "mild diabetic retinopathy",
    2: "moderate diabetic retinopathy",
    3: "severe diabetic retinopathy",
    4: "proliferative diabetic retinopathy",
}


def label_to_disease(label_id: int) -> str:
    return RETINAMNIST_DISEASE_NAMES.get(label_id, f"diabetic retinopathy grade {label_id}")


def _clean_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def neutralize_caption(text: str) -> str:
    # Remove shape/texture/color words to reduce bias.
    words = SHAPE_WORDS | TEXTURE_WORDS | COLOR_WORDS
    if words:
        pattern = r"\b(" + "|".join(map(re.escape, words)) + r")\b"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return _clean_caption(text)


COLOR_PALETTE = {
    "red": (200, 50, 50),
    "green": (50, 160, 50),
    "blue": (60, 80, 180),
    "yellow": (220, 200, 60),
    "orange": (230, 140, 40),
    "brown": (150, 90, 50),
    "gray": (130, 130, 130),
}


def dominant_color_name(image: Image.Image) -> str:
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return "gray"
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    spread = mean_rgb.max() - mean_rgb.min()
    if spread < 15:
        return "gray"
    # Choose closest color in palette (exclude gray for non-gray case)
    best_name = "red"
    best_dist = float("inf")
    for name, rgb in COLOR_PALETTE.items():
        if name == "gray":
            continue
        rgb = np.array(rgb, dtype=np.float32)
        dist = float(np.sum((mean_rgb - rgb) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def average_luminance(image: Image.Image) -> float:
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0
    # Standard luma transform
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(luma.mean())


def generate_captions(
    root: str,
    split: str,
    backend: str,
    device: str,
    max_samples: Optional[int],
    style: str,
    neutral_mode: str,
    color_mode: str,
    brightness_threshold: float,
    append_label: bool,
    log_every: int,
    mix_generic: bool,
    generic_prob: float,
    seed: int,
) -> List[Dict]:
    dataset = load_dataset(root, split)
    total = len(dataset)

    records: List[Dict] = []
    batch: List[Image.Image] = []
    batch_indices: List[int] = []
    batch_labels: List[int] = []
    last_log = 0
    rng = np.random.default_rng(seed)

    processor = model = torch_mod = None
    if backend in {"blip", "blip2"} and not (
        (style == "neutral" and neutral_mode == "template")
        or style == "color"
    ):
        if backend == "blip":
            processor, model, torch_mod = init_blip(device)
        else:
            processor, model, torch_mod = init_blip2(device)

    def flush_batch():
        nonlocal last_log
        if not batch:
            return
        if style == "color":
            if color_mode == "dark":
                captions = ["a retinal fundus image with dominant dark tones" for _ in batch]
            elif color_mode == "bright":
                captions = ["a retinal fundus image with dominant bright tones" for _ in batch]
            elif color_mode == "by_brightness":
                captions = []
                for img in batch:
                    lum = average_luminance(img)
                    tone = "dark" if lum < brightness_threshold else "bright"
                    captions.append(f"a retinal fundus image with dominant {tone} tones")
            else:
                captions = [f"a retinal fundus image with dominant {dominant_color_name(img)} tones" for img in batch]
            if append_label:
                captions = [
                    f"{cap} showing {label_to_disease(label)}"
                    for cap, label in zip(captions, batch_labels)
                ]
        elif style == "neutral" and neutral_mode == "template":
            captions = [NEUTRAL_TEMPLATES[i % len(NEUTRAL_TEMPLATES)] for i in batch_indices]
        elif style == "neutral" and neutral_mode == "label":
            captions = [f"retinal fundus image showing {label_to_disease(label)}" for label in batch_labels]
        else:
            if backend == "blip":
                captions = caption_with_blip(batch, device=device, processor=processor, model=model, torch_mod=torch_mod)
            elif backend == "blip2":
                captions = caption_with_blip2(batch, device=device, processor=processor, model=model, torch_mod=torch_mod)
            elif backend == "dummy":
                captions = caption_dummy(batch)
            else:
                raise ValueError(f"Unknown backend '{backend}'.")

        if style == "neutral":
            if neutral_mode == "template":
                captions = [NEUTRAL_TEMPLATES[i % len(NEUTRAL_TEMPLATES)] for i in batch_indices]
            elif neutral_mode == "label":
                captions = [f"retinal fundus image showing {label_to_disease(label)}" for label in batch_labels]
            elif neutral_mode == "strip":
                captions = [neutralize_caption(c) for c in captions]
            else:
                raise ValueError(f"Unknown neutral_mode '{neutral_mode}'.")

        for idx, caption in zip(batch_indices, captions):
            if mix_generic and rng.random() < generic_prob:
                caption = GENERIC_TEMPLATES[idx % len(GENERIC_TEMPLATES)]
            records.append(
                {
                    "split": split,
                    "index": idx,
                    "caption": caption.strip(),
                }
            )

        batch.clear()
        batch_indices.clear()
        batch_labels.clear()

        if log_every > 0 and (len(records) - last_log) >= log_every:
            print(f"Generated {len(records)} captions...")
            last_log = len(records)

    for idx in iter_indices(total, max_samples):
        image, target = dataset[idx]
        if image.mode != "RGB":
            image = image.convert("RGB")
        batch.append(image)
        batch_indices.append(idx)
        try:
            label_id = int(target[0])
        except Exception:
            label_id = int(target)
        batch_labels.append(label_id)

        if len(batch) >= 8:
            flush_batch()

    flush_batch()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RetinaMNIST captions with a captioning model")
    parser.add_argument("--root", type=str, default="./dataset/tomburgert")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--backend", type=str, default="blip", choices=["blip", "blip2", "dummy"])
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=str, default="./dataset/retinamnist_captions.jsonl")
    parser.add_argument(
        "--style",
        type=str,
        default="neutral",
        choices=["neutral", "raw", "color"],
        help="neutral: remove cues; raw: keep original caption; color: add dominant color",
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        default="auto",
        choices=["auto", "dark", "bright", "by_brightness"],
        help="For style=color: auto uses dominant color; dark/bright force a bias; by_brightness chooses dark/bright per image.",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=110.0,
        help="Luminance threshold (0-255) used by --color-mode by_brightness.",
    )
    parser.add_argument(
        "--append-label",
        action="store_true",
        help="For style=color: append disease name based on RetinaMNIST label.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N captions. Set to 0 to disable.",
    )
    parser.add_argument(
        "--mix-generic",
        action="store_true",
        help="Mix in generic retina captions to preserve alignment with general prompts.",
    )
    parser.add_argument(
        "--generic-prob",
        type=float,
        default=0.3,
        help="Probability of replacing a caption with a generic template when --mix-generic is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for generic caption mixing.",
    )
    parser.add_argument(
        "--neutral-mode",
        type=str,
        default="template",
        choices=["template", "strip", "label"],
        help="template: neutral templates; strip: remove cue words from BLIP; label: neutral grade-based caption",
    )
    args = parser.parse_args()

    records = generate_captions(
        root=args.root,
        split=args.split,
        backend=args.backend,
        device=args.device,
        max_samples=args.max_samples,
        style=args.style,
        neutral_mode=args.neutral_mode,
        color_mode=args.color_mode,
        brightness_threshold=args.brightness_threshold,
        append_label=args.append_label,
        log_every=args.log_every,
        mix_generic=args.mix_generic,
        generic_prob=args.generic_prob,
        seed=args.seed,
    )

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} captions to {args.out}")


if __name__ == "__main__":
    main()
