#!/usr/bin/env python3
"""
Download a subset of COCO train2017 images and captions.

Example:
  python3 scripts/download_coco_subset.py \
    --out-dir ./dataset/coco \
    --num-images 15000 \
    --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
import zipfile
from collections import defaultdict
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_BASE_URL = "http://images.cocodataset.org/train2017"


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None
        chunk_size = 1024 * 1024
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                pbar.update(len(chunk))
    os.replace(tmp, dest)


def _download_image(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        out.write(response.read())
    os.replace(tmp, dest)


def _ensure_annotations(out_dir: str) -> str:
    ann_zip = os.path.join(out_dir, "annotations_trainval2017.zip")
    ann_dir = os.path.join(out_dir, "annotations")
    ann_json = os.path.join(ann_dir, "captions_train2017.json")

    if not os.path.exists(ann_json):
        _download(ANNOTATIONS_URL, ann_zip)
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extractall(out_dir)
    return ann_json


def load_coco_captions(ann_json: str) -> Dict[int, List[str]]:
    with open(ann_json, "r") as f:
        data = json.load(f)
    caps = defaultdict(list)
    for ann in data["annotations"]:
        caps[ann["image_id"]].append(ann["caption"])
    return caps, data["images"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO train2017 subset")
    parser.add_argument("--out-dir", type=str, default="./dataset/coco")
    parser.add_argument("--num-images", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--captions-per-image", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ann_json = _ensure_annotations(args.out_dir)
    caps_map, images = load_coco_captions(ann_json)

    # Sample images
    rng = random.Random(args.seed)
    eligible = [img for img in images if img["id"] in caps_map]
    if args.num_images > len(eligible):
        raise ValueError(f"Requested {args.num_images} images, but only {len(eligible)} available.")
    sample = rng.sample(eligible, args.num_images)

    images_dir = os.path.join(args.out_dir, "train2017")
    os.makedirs(images_dir, exist_ok=True)

    # Download images
    futures = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for img in sample:
            file_name = img["file_name"]
            url = f"{IMAGES_BASE_URL}/{file_name}"
            dest = os.path.join(images_dir, file_name)
            if os.path.exists(dest):
                continue
            futures.append(executor.submit(_download_image, url, dest))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            pass

    # Write subset captions
    subset_path = os.path.join(args.out_dir, "coco_train2017_subset.jsonl")
    with open(subset_path, "w") as f:
        for img in sample:
            image_id = img["id"]
            file_name = img["file_name"]
            captions = caps_map[image_id]
            rng.shuffle(captions)
            for cap in captions[: args.captions_per_image]:
                rec = {
                    "split": "train2017",
                    "image_id": image_id,
                    "image_path": os.path.join(images_dir, file_name),
                    "caption": cap,
                }
                f.write(json.dumps(rec) + "\n")

    print(f"Saved subset captions to {subset_path}")


if __name__ == "__main__":
    main()
