"""
Text + Image alignment utilities for feature reliance experiments.

This module provides:
1. Caption content manipulation (shape/texture/color/neutral emphasis)
2. Prompt engineering helpers for inference-time probing
3. Text embedding space axis analysis (shape vs texture vs color)
4. Caption augmentation / style transfer (rule-based with optional LLM hook)
5. Token-level importance analysis (leave-one-out masking)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re

import torch


# ----------------------------
# 1) Caption Content Manipulation
# ----------------------------

COLOR_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "gold", "silver", "beige", "tan", "maroon", "cyan", "magenta",
}

TEXTURE_WORDS = {
    "smooth", "rough", "furry", "hairy", "striped", "spotted", "speckled", "grainy", "matte",
    "glossy", "shiny", "wrinkled", "bumpy", "soft", "hard", "velvety", "patchy",
}

SHAPE_WORDS = {
    "round", "circular", "oval", "square", "rectangular", "triangular", "long", "short",
    "curved", "straight", "thin", "thick", "wide", "narrow", "branching", "elongated",
}


def _clean_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def _remove_words(text: str, words: Iterable[str]) -> str:
    if not words:
        return text
    pattern = r"\b(" + "|".join(map(re.escape, words)) + r")\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def focus_caption(caption: str, focus: str) -> str:
    """
    Focus caption content on a particular feature type by pruning other cues
    and injecting a focus prefix.
    """
    focus = focus.lower()
    text = caption

    if focus == "shape":
        text = _remove_words(text, COLOR_WORDS | TEXTURE_WORDS)
        prefix = "shape-focused description: "
    elif focus == "texture":
        text = _remove_words(text, COLOR_WORDS | SHAPE_WORDS)
        prefix = "texture-focused description: "
    elif focus == "color":
        text = _remove_words(text, TEXTURE_WORDS | SHAPE_WORDS)
        prefix = "color-focused description: "
    elif focus == "neutral":
        prefix = ""
    else:
        raise ValueError(f"Unknown focus '{focus}'. Use shape/texture/color/neutral.")

    return _clean_caption(prefix + text)


def make_attribute_caption(
    subject: str,
    shape: Optional[str] = None,
    texture: Optional[str] = None,
    color: Optional[str] = None,
    focus: str = "neutral",
) -> str:
    """
    Build a caption from structured attributes, then optionally focus it.
    """
    parts = [subject]
    if shape:
        parts.append(f"with {shape} shape")
    if texture:
        parts.append(f"with {texture} texture")
    if color:
        parts.append(f"in {color} color")
    caption = " ".join(parts)
    return focus_caption(caption, focus)


# ----------------------------
# 2) Prompt Engineering Helpers
# ----------------------------

PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "clip_basic": [
        "a photo of {label}",
        "a close-up photo of {label}",
        "a cropped photo of {label}",
        "a bright photo of {label}",
    ],
    "shape": [
        "the shape of {label}",
        "an outline of {label}",
        "a silhouette of {label}",
        "a shape-focused view of {label}",
    ],
    "texture": [
        "the texture of {label}",
        "a close-up texture of {label}",
        "a texture-focused view of {label}",
        "surface details of {label}",
    ],
    "color": [
        "the color of {label}",
        "a color-focused view of {label}",
        "dominant colors of {label}",
        "color palette of {label}",
    ],
}


def build_prompts(label: str, prompt_set: str = "clip_basic") -> List[str]:
    if prompt_set not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt_set '{prompt_set}'.")
    return [t.format(label=label) for t in PROMPT_TEMPLATES[prompt_set]]


# ----------------------------
# 3) Text Embedding Space Axis
# ----------------------------

@dataclass
class EmbeddingAxis:
    name: str
    direction: torch.Tensor  # normalized vector
    anchors: Tuple[str, str]


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def build_alignment_axis(
    clip_backend: "ClipBackend",
    positive_prompt: str,
    negative_prompt: str,
    name: str = "axis",
) -> EmbeddingAxis:
    text_emb = clip_backend.encode_text([positive_prompt, negative_prompt])
    axis = normalize(text_emb[0] - text_emb[1])
    return EmbeddingAxis(name=name, direction=axis, anchors=(positive_prompt, negative_prompt))


def project_embeddings(embeddings: torch.Tensor, axis: EmbeddingAxis) -> torch.Tensor:
    embeddings = normalize(embeddings)
    return (embeddings @ axis.direction).squeeze(-1)


# ----------------------------
# 4) Caption Augmentation / Style Transfer
# ----------------------------

class CaptionRewriter:
    def rewrite(self, caption: str, register: str, focus: str = "neutral") -> str:
        raise NotImplementedError


class RuleBasedRewriter(CaptionRewriter):
    """
    Light-weight, dependency-free style transfer.
    For LLM-based rewriting, plug in a different CaptionRewriter.
    """

    REGISTER_PREFIX = {
        "scientific": "scientific description:",
        "clinical": "clinical description:",
        "casual": "casual caption:",
        "formal": "formal description:",
    }

    def rewrite(self, caption: str, register: str, focus: str = "neutral") -> str:
        register = register.lower()
        prefix = self.REGISTER_PREFIX.get(register, "description:")
        focused = focus_caption(caption, focus)
        return _clean_caption(f"{prefix} {focused}")


def augment_coco_captions(
    captions_json_path: str,
    output_path: str,
    registers: Sequence[str],
    focuses: Sequence[str],
    rewriter: Optional[CaptionRewriter] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Load MSCOCO captions JSON and write rewritten variants.
    Output JSON format: list of {image_id, caption, register, focus}.
    """
    if rewriter is None:
        rewriter = RuleBasedRewriter()

    with open(captions_json_path, "r") as f:
        coco = json.load(f)

    annotations = coco.get("annotations", [])
    if limit is not None:
        annotations = annotations[:limit]

    out = []
    for ann in annotations:
        caption = ann["caption"]
        image_id = ann["image_id"]
        for register in registers:
            for focus in focuses:
                rewritten = rewriter.rewrite(caption, register=register, focus=focus)
                out.append(
                    {
                        "image_id": image_id,
                        "caption": rewritten,
                        "register": register,
                        "focus": focus,
                        "source": caption,
                    }
                )

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


# ----------------------------
# 5) Token-Level Analysis (Leave-One-Out)
# ----------------------------

@dataclass
class TokenAttribution:
    token: str
    token_id: int
    importance: float


def token_level_importance(
    clip_backend: "ClipBackend",
    image_embedding: torch.Tensor,
    caption: str,
    pad_token_id: int = 0,
) -> List[TokenAttribution]:
    """
    Token-level importance using leave-one-out masking.
    importance = base_similarity - masked_similarity
    """
    device = clip_backend.device
    tokens = clip_backend.tokenize([caption]).to(device)
    base_text = clip_backend.encode_text_tokens(tokens)
    base_sim = (normalize(image_embedding) * normalize(base_text)).sum(dim=-1).item()

    attributions: List[TokenAttribution] = []
    tokens_np = tokens[0].tolist()
    token_strs = clip_backend.decode_tokens(tokens_np)

    # Skip special BOS/EOS if present (assume first and last are special)
    for idx in range(1, len(tokens_np) - 1):
        masked = tokens.clone()
        masked[0, idx] = pad_token_id
        masked_text = clip_backend.encode_text_tokens(masked)
        masked_sim = (normalize(image_embedding) * normalize(masked_text)).sum(dim=-1).item()
        importance = base_sim - masked_sim

        token_str = token_strs[idx] if idx < len(token_strs) else str(tokens_np[idx])
        attributions.append(
            TokenAttribution(token=token_str, token_id=tokens_np[idx], importance=importance)
        )

    return attributions


# ----------------------------
# CLIP Backend Abstraction
# ----------------------------

class ClipBackend:
    """
    Minimal interface for CLIP-like models.
    Implementations must provide: encode_text, encode_text_tokens, encode_image, tokenize, decode_tokens, device
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def tokenize(self, prompts: Sequence[str]) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def decode_tokens(self, token_ids: Sequence[int]) -> List[str]:  # pragma: no cover
        return [str(t) for t in token_ids]


class OpenClipBackend(ClipBackend):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        try:
            import open_clip
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "open_clip is required for OpenClipBackend. Install with `pip install open_clip_torch`."
            ) from exc

        self.open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def tokenize(self, prompts: Sequence[str]) -> torch.Tensor:
        return self.tokenizer(prompts)

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenize(prompts).to(self.device)
        return self.encode_text_tokens(tokens)

    def encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_text(tokens)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_image(images.to(self.device))

    def decode_tokens(self, token_ids: Sequence[int]) -> List[str]:
        if hasattr(self.tokenizer, "decode"):
            return [self.tokenizer.decode([t]) for t in token_ids]
        return super().decode_tokens(token_ids)


# ----------------------------
# Convenience Probing Functions
# ----------------------------

def score_image_with_prompts(
    clip_backend: ClipBackend,
    image_tensor: torch.Tensor,
    prompts: Sequence[str],
) -> Dict[str, float]:
    image_emb = clip_backend.encode_image(image_tensor.unsqueeze(0))
    text_embs = clip_backend.encode_text(prompts)
    image_emb = normalize(image_emb)
    text_embs = normalize(text_embs)
    sims = (image_emb @ text_embs.T).squeeze(0)
    return {p: float(s) for p, s in zip(prompts, sims)}


def image_axis_projection(
    clip_backend: ClipBackend,
    image_tensor: torch.Tensor,
    positive_prompt: str,
    negative_prompt: str,
    axis_name: str = "axis",
) -> float:
    axis = build_alignment_axis(
        clip_backend, positive_prompt=positive_prompt, negative_prompt=negative_prompt, name=axis_name
    )
    img_emb = clip_backend.encode_image(image_tensor.unsqueeze(0))
    proj = project_embeddings(img_emb, axis)
    if proj.ndim == 0:
        return float(proj.item())
    return float(proj.squeeze()[0].item())
