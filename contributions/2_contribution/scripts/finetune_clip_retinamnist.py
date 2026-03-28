#!/usr/bin/env python3
"""
Fine-tune a CLIP model on RetinaMNIST image-caption pairs.

Prereqs:
  pip install open_clip_torch

Example:
  python3 scripts/finetune_clip_retinamnist.py \
    --root ./dataset/tomburgert \
    --captions ./dataset/retinamnist_captions.jsonl \
    --epochs 5 \
    --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from medmnist import RetinaMNIST


@dataclass
class CaptionRecord:
    split: str
    index: int
    caption: str


class RetinaCaptionDataset(Dataset):
    def __init__(
        self,
        root: str,
        caption_file: str,
        split: str,
        preprocess,
        max_samples: Optional[int] = None,
        caption_key: str = "caption",
        use_all_splits: bool = False,
        preview: int = 0,
    ):
        self.root = root
        self.preprocess = preprocess
        self.dataset = RetinaMNIST(root=root, split=split, download=True, size=224)
        self.records = self._load_records(
            caption_file,
            split,
            max_samples=max_samples,
            caption_key=caption_key,
            use_all_splits=use_all_splits,
        )
        if preview > 0:
            for rec in self.records[:preview]:
                print(f"Caption preview: index={rec.index} split={rec.split} caption={rec.caption}")

    @staticmethod
    def _load_records(
        path: str,
        split: str,
        max_samples: Optional[int],
        caption_key: str,
        use_all_splits: bool,
    ) -> List[CaptionRecord]:
        records: List[CaptionRecord] = []
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if not use_all_splits and obj.get("split") != split:
                    continue
                caption = obj.get(caption_key, obj.get("caption"))
                if caption is None:
                    continue
                rec_split = obj.get("split", split)
                records.append(CaptionRecord(split=rec_split, index=obj["index"], caption=caption))
                if max_samples is not None and len(records) >= max_samples:
                    break
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image, _ = self.dataset[rec.index]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.preprocess(image)
        return image, rec.caption


def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale * image_features @ text_features.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on RetinaMNIST captions")
    parser.add_argument("--root", type=str, default="./dataset/tomburgert")
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--caption-key", type=str, default="caption")
    parser.add_argument("--use-all-splits", action="store_true")
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="./logs/clip_retinamnist.pt")
    parser.add_argument("--freeze-image", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")
    args = parser.parse_args()

    try:
        import open_clip
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "open_clip_torch is required. Install with `pip install open_clip_torch`."
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.train()

    if args.freeze_image:
        for p in model.visual.parameters():
            p.requires_grad = False
    if args.freeze_text:
        for p in model.transformer.parameters():
            p.requires_grad = False

    dataset = RetinaCaptionDataset(
        root=args.root,
        caption_file=args.captions,
        split=args.split,
        preprocess=preprocess,
        max_samples=args.max_samples,
        caption_key=args.caption_key,
        use_all_splits=args.use_all_splits,
        preview=args.preview,
    )
    if len(dataset) == 0:
        raise ValueError("No caption records found for the selected split. Check --captions and --split.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.to(args.device)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, captions in loader:
            images = images.to(args.device)
            tokens = tokenizer(list(captions)).to(args.device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
            logit_scale = model.logit_scale.exp()

            loss = clip_loss(image_features, text_features, logit_scale)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.output)
    print(f"Saved fine-tuned model to {args.output}")


if __name__ == "__main__":
    main()
