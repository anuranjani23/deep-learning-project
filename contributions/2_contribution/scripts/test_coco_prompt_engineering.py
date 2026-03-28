#!/usr/bin/env python3
"""
Test prompt engineering relevance on a COCO subset using a pretrained CLIP backbone.

Requires:
  pip install open_clip_torch

Example:
  python3 scripts/test_coco_prompt_engineering.py \
    --captions ./dataset/coco/coco_train2017_subset.jsonl \
    --max-samples 200 \
    --prompt-sets clip_basic,shape,texture,color
"""

from __future__ import annotations

import argparse
import json
import csv
import os
import random
import sys
from typing import Dict, List

from PIL import Image
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from text_experiments import OpenClipBackend, build_prompts, normalize


STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "with", "and", "to", "for", "by", "from",
    "is", "are", "was", "were", "this", "that", "these", "those", "at", "as",
}


def extract_noun_label(caption: str) -> str:
    # Very lightweight heuristic: pick the last non-stopword token.
    tokens = []
    word = ""
    for ch in caption.lower():
        if ch.isalnum() or ch == "-":
            word += ch
        else:
            if word:
                tokens.append(word)
                word = ""
    if word:
        tokens.append(word)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens[-1] if tokens else caption


def extract_short_label(caption: str, max_tokens: int = 3) -> str:
    # Simple heuristic: take first N non-stopword tokens to keep some context.
    tokens = []
    word = ""
    for ch in caption.lower():
        if ch.isalnum() or ch == "-":
            word += ch
        else:
            if word:
                tokens.append(word)
                word = ""
    if word:
        tokens.append(word)
    tokens = [t for t in tokens if t not in STOPWORDS]
    if not tokens:
        return caption
    return " ".join(tokens[:max_tokens])


def load_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            image_path = obj.get("image_path")
            caption = obj.get("caption")
            if not image_path or caption is None:
                continue
            records.append({"image_path": image_path, "caption": caption})
    return records


def score_caption(clip: OpenClipBackend, image_emb: torch.Tensor, caption: str) -> float:
    text_emb = clip.encode_text([caption])
    image_emb = normalize(image_emb)
    text_emb = normalize(text_emb)
    return float((image_emb @ text_emb.T).squeeze(0).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO prompt engineering test (pretrained CLIP)")
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--prompt-sets",
        type=str,
        default="clip_basic,shape,texture,color",
        help="Comma-separated prompt sets: clip_basic,shape,texture,color",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--clip-checkpoint", type=str, default=None, help="Path to fine-tuned CLIP checkpoint")
    parser.add_argument("--compare-base", action="store_true", help="Compare fine-tuned vs base CLIP")
    parser.add_argument("--label-mode", type=str, default="caption", choices=["caption", "noun", "short"])
    parser.add_argument("--show-examples", type=int, default=0)
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--preset", type=str, default="none", choices=["none", "best"])
    args = parser.parse_args()

    if args.preset == "best":
        args.prompt_sets = "clip_basic,shape,texture,color"
        args.label_mode = "caption"
        if args.max_samples == 200:
            args.max_samples = 500
        if args.show_examples == 0:
            args.show_examples = 3
        if args.save_csv is None:
            args.save_csv = "./logs/coco_prompt_report.csv"
        print("Using preset=best (caption labels, 4 prompt sets, examples, CSV).")

    records = load_records(args.captions)
    if not records:
        raise ValueError("No records found in captions file.")

    rng = random.Random(args.seed)
    if args.max_samples is not None and args.max_samples < len(records):
        records = rng.sample(records, args.max_samples)

    prompt_sets = [p.strip() for p in args.prompt_sets.split(",") if p.strip()]
    clip = OpenClipBackend(device=args.device)
    if args.clip_checkpoint:
        import torch
        state = torch.load(args.clip_checkpoint, map_location=clip.device, weights_only=True)
        state_dict = state.get("state_dict", state)
        missing, unexpected = clip.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)}")

    base_clip = None
    if args.compare_base and args.clip_checkpoint:
        base_clip = OpenClipBackend(device=args.device)

    totals = {ps: 0.0 for ps in prompt_sets}
    deltas = {ps: 0.0 for ps in prompt_sets}
    base_total = 0.0

    skipped = 0
    progress = tqdm(records, desc="Scoring COCO prompts", unit="image")
    for i, rec in enumerate(progress, 1):
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            skipped += 1
            progress.set_postfix(skipped=skipped)
            continue
        image_tensor = clip.preprocess(image)
        image_emb = clip.encode_image(image_tensor.unsqueeze(0))

        base_score = score_caption(clip, image_emb, rec["caption"])
        base_score_ref = base_score
        if base_clip is not None:
            base_score_ref = score_caption(base_clip, image_emb, rec["caption"])
        base_total += base_score

        label = rec["caption"]
        if args.label_mode == "noun":
            label = extract_noun_label(label)
        elif args.label_mode == "short":
            label = extract_short_label(label)
        for ps in prompt_sets:
            prompts = build_prompts(label, ps)
            scores = []
            for p in prompts:
                scores.append(score_caption(clip, image_emb, p))
            avg = sum(scores) / max(len(scores), 1)
            totals[ps] += avg
            deltas[ps] += (avg - base_score_ref)

        progress.set_postfix(skipped=skipped, base=f"{base_total / i:.4f}")

        if args.show_examples > 0 and i <= args.show_examples:
            print(f"\nExample {i}")
            print(f"  caption: {rec['caption']}")
            print(f"  label:   {label}")
            print(f"  base:    {base_score:.4f}")
            for ps in prompt_sets:
                avg = (totals[ps] / i) if i > 0 else 0.0
                print(f"  {ps}:   {avg:.4f}")

    n = len(records) - skipped
    if n <= 0:
        raise ValueError("No readable images found. Download may still be running or files are corrupted.")
    print("\nAverage similarity (prompt set):")
    for ps in prompt_sets:
        print(f"  {ps}: {totals[ps] / n:.4f}")
    print(f"  base_caption: {base_total / n:.4f}")

    print("\nAverage delta vs base caption (prompt set):")
    for ps in prompt_sets:
        print(f"  {ps}: {deltas[ps] / n:+.4f}")

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            fieldnames = ["image_path", "caption", "label", "base_score"] + [f"{ps}_avg" for ps in prompt_sets]
            if base_clip is not None:
                fieldnames += ["base_caption_score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Re-run to write per-sample rows
            csv_progress = tqdm(records[:n], desc="Writing CSV", unit="image")
            for rec in csv_progress:
                try:
                    image = Image.open(rec["image_path"]).convert("RGB")
                except Exception:
                    continue
                image_tensor = clip.preprocess(image)
                image_emb = clip.encode_image(image_tensor.unsqueeze(0))
                base_score = score_caption(clip, image_emb, rec["caption"])
                base_caption_score = None
                if base_clip is not None:
                    base_caption_score = score_caption(base_clip, image_emb, rec["caption"])
                label = rec["caption"]
                if args.label_mode == "noun":
                    label = extract_noun_label(label)
                elif args.label_mode == "short":
                    label = extract_short_label(label)
                row = {
                    "image_path": rec["image_path"],
                    "caption": rec["caption"],
                    "label": label,
                    "base_score": base_score,
                }
                if base_clip is not None:
                    row["base_caption_score"] = base_caption_score
                for ps in prompt_sets:
                    prompts = build_prompts(label, ps)
                    scores = [score_caption(clip, image_emb, p) for p in prompts]
                    row[f"{ps}_avg"] = sum(scores) / max(len(scores), 1)
                writer.writerow(row)


if __name__ == "__main__":
    main()
