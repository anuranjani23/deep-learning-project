#!/usr/bin/env python3
"""
Quick test script for text_experiments.py on RetinaMNIST.

Example:
  python3 scripts/test_retinamnist_text.py --root ./dataset/tomburgert --index 0 --prompt-set shape
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from typing import Any, Dict

import torch
from PIL import Image

from medmnist import INFO, RetinaMNIST

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from text_experiments import (
    OpenClipBackend,
    build_prompts,
    image_axis_projection,
    score_image_with_prompts,
    token_level_importance,
)


def _get_label_name(label_id: int) -> str:
    label_map: Any = INFO["retinamnist"]["label"]
    if isinstance(label_map, dict):
        return label_map.get(str(label_id), label_map.get(label_id, str(label_id)))
    if isinstance(label_map, (list, tuple)) and label_id < len(label_map):
        return str(label_map[label_id])
    return str(label_id)


def load_sample(root: str, split: str, index: int) -> Dict[str, Any]:
    dataset = RetinaMNIST(root=root, split=split, download=True, size=224)
    image, target = dataset[index]
    if image.mode != "RGB":
        image = image.convert("RGB")
    label_id = int(target[0])
    label_name = _get_label_name(label_id)
    return {
        "image": image,
        "label_id": label_id,
        "label_name": label_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinaMNIST text experiment demo")
    parser.add_argument("--root", type=str, default="./dataset/tomburgert")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prompt-set", type=str, default="shape", choices=["clip_basic", "shape", "texture", "color"])
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--label", type=str, default=None, help="Override label text used for prompts")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--clip-checkpoint", type=str, default=None, help="Path to fine-tuned CLIP checkpoint")
    parser.add_argument("--compare-base", action="store_true", help="Compare fine-tuned vs base CLIP")
    args = parser.parse_args()

    sample = load_sample(args.root, args.split, args.index)
    image: Image.Image = sample["image"]
    label_name = sample["label_name"]

    clip = OpenClipBackend(device=args.device)
    if args.clip_checkpoint:
        import torch
        state = torch.load(args.clip_checkpoint, map_location=clip.device, weights_only=True)
        state_dict = state.get("state_dict", state)
        missing, unexpected = clip.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)}")
    image_tensor = clip.preprocess(image)

    if args.label:
        prompt_label = args.label
    else:
        # If the dataset provides numeric labels, fall back to a semantic prompt label
        prompt_label = label_name
        if isinstance(prompt_label, str) and prompt_label.strip().isdigit():
            prompt_label = "retinal fundus image"

    prompts = build_prompts(prompt_label, args.prompt_set)
    scores = score_image_with_prompts(clip, image_tensor, prompts)

    print("Sample label:", label_name)
    print("Prompt label:", prompt_label)
    if args.compare_base and args.clip_checkpoint:
        base_clip = OpenClipBackend(device=args.device)
        base_scores = score_image_with_prompts(base_clip, image_tensor, prompts)
        print("Prompt scores (finetuned | base | delta):")
        for p, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            b = base_scores.get(p, 0.0)
            print(f"  {p}: {s:.4f} | {b:.4f} | {s - b:+.4f}")
    else:
        print("Prompt scores:")
        for p, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {p}: {s:.4f}")

    axis_score = image_axis_projection(
        clip,
        image_tensor,
        positive_prompt="branching vessel structure in a retinal fundus image",
        negative_prompt="patchy lesion texture in a retinal fundus image",
        axis_name="shape_vs_texture",
    )
    if args.compare_base and args.clip_checkpoint:
        base_axis = image_axis_projection(
            base_clip,
            image_tensor,
            positive_prompt="branching vessel structure in a retinal fundus image",
            negative_prompt="patchy lesion texture in a retinal fundus image",
            axis_name="shape_vs_texture",
        )
        print(f"Axis projection (shape vs texture): {axis_score:.4f} | {base_axis:.4f} | {axis_score - base_axis:+.4f}")
    else:
        print(f"Axis projection (shape vs texture): {axis_score:.4f}")

    caption = args.caption or f"a retinal fundus image of {label_name} with branching vessels"
    image_emb = clip.encode_image(image_tensor.unsqueeze(0))
    attributions = token_level_importance(clip, image_emb, caption)
    # Filter out special tokens and punctuation-only tokens
    filtered = []
    for attr in attributions:
        token = attr.token.strip()
        if token.startswith("<") and token.endswith(">"):
            continue
        if not re.search(r"[A-Za-z0-9]", token):
            continue
        filtered.append(attr)
    attributions = sorted(filtered, key=lambda x: x.importance, reverse=True)[: args.topk]

    if args.compare_base and args.clip_checkpoint:
        base_image_emb = base_clip.encode_image(image_tensor.unsqueeze(0))
        base_attrs = token_level_importance(base_clip, base_image_emb, caption)
        base_map = {a.token: a.importance for a in base_attrs}
        print("Top token importances (finetuned | base | delta):")
        for attr in attributions:
            b = base_map.get(attr.token, 0.0)
            print(f"  {attr.token}: {attr.importance:.4f} | {b:.4f} | {attr.importance - b:+.4f}")
    else:
        print("Top token importances:")
        for attr in attributions:
            print(f"  {attr.token}: {attr.importance:.4f}")


if __name__ == "__main__":
    main()
