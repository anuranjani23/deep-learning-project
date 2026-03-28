#!/usr/bin/env python3
"""
Download AID dataset from Hugging Face and store in the folder layout
expected by AIDDataset: <out_dir>/data/<class_name>/*.jpg

Requires:
  pip install datasets
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset, DownloadConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AID dataset from Hugging Face")
    parser.add_argument("--out-dir", type=str, default="./dataset/tomburgert")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token to avoid rate limits")
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--num-proc", type=int, default=1)
    args = parser.parse_args()

    out_root = os.path.join(args.out_dir, "data")
    os.makedirs(out_root, exist_ok=True)

    download_config = DownloadConfig(
        max_retries=args.max_retries,
        num_proc=args.num_proc,
    )
    ds = load_dataset("blanchon/AID", token=args.hf_token, download_config=download_config)
    splits = ds.keys()
    for split in splits:
        for item in ds[split]:
            image = item["image"]
            label = item.get("label", "unknown")
            label_name = str(label)
            class_dir = os.path.join(out_root, label_name)
            os.makedirs(class_dir, exist_ok=True)
            fname = item.get("image_id", None)
            if fname is None:
                fname = f"{split}_{item.get('id', 0)}.jpg"
            if not str(fname).lower().endswith(".jpg"):
                fname = f"{fname}.jpg"
            path = os.path.join(class_dir, fname)
            if not os.path.exists(path):
                image.save(path)

    print(f"Saved AID images to {out_root}")


if __name__ == "__main__":
    main()
