"""
Shared image-composition and prompt construction for SatQuery AI.

This module is imported by BOTH the training scripts and the inference
registry. That is deliberate and load-bearing: if training builds its
two-image composites one way and inference builds them another, the model is
served inputs it never saw, and accuracy silently collapses with no error
anywhere. Every composite and every prompt string in the system comes from
here.

Why aspect ratio matters
------------------------
PaliGemma-3b-pt-224 resizes whatever it is given to a fixed 224x224 square.
Pasting two square scenes side by side into a 2W x H canvas and handing that
to the processor squashes each scene by 2:1 horizontally - the model sees
distorted geometry that does not match its pretraining. Instead we fit each
scene into its own square cell and letterbox the result, so shapes are
preserved. Effective per-scene resolution is the same either way; only the
distortion differs.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from PIL import Image

Layout = Literal["horizontal", "vertical"]

# Neutral mid-grey letterbox. Black would read as nodata/water to an
# RS-adapted model; mid-grey is closer to the dataset mean.
PAD_COLOR = (128, 128, 128)

# Canonical prompt prefixes. Changing any string here changes what the model
# must be trained on - keep training and inference in lockstep by importing.
CHANGE_PREFIX = (
    "The left image is from time T1, the right image is from time T2. "
    "Answer the question about what changed: "
)
FUSION_PREFIX = (
    "The left image is optical/multispectral, the right image is SAR radar "
    "backscatter in decibels. Use both together to answer: "
)


def make_pair_composite(
    img1: Image.Image,
    img2: Image.Image,
    size: int = 224,
    layout: Layout = "horizontal",
    pad_color: Tuple[int, int, int] = PAD_COLOR,
) -> Image.Image:
    """
    Combine two images into one square `size` x `size` canvas without
    distorting either.

    Each image is fitted into a (size//2) x (size//2) cell preserving its own
    aspect ratio, then the two cells are placed side by side (or stacked) and
    centred on the square canvas.

    Parameters
    ----------
    img1, img2 : PIL.Image
        Already loaded as RGB (use modules.raster_io.load_as_rgb).
    size : int
        Edge length of the output square. Match the model's input resolution
        (224 for paligemma-3b-pt-224, 448 for pt-448).
    layout : "horizontal" | "vertical"
        "horizontal" puts img1 on the left - this is what the CHANGE_PREFIX
        and FUSION_PREFIX strings describe, so do not change it without
        changing those strings and retraining.
    """
    cell = size // 2
    canvas = Image.new("RGB", (size, size), pad_color)

    a = _fit_into(img1.convert("RGB"), cell, pad_color)
    b = _fit_into(img2.convert("RGB"), cell, pad_color)

    if layout == "horizontal":
        y = (size - cell) // 2
        canvas.paste(a, (0, y))
        canvas.paste(b, (cell, y))
    else:
        x = (size - cell) // 2
        canvas.paste(a, (x, 0))
        canvas.paste(b, (x, cell))

    return canvas


def _fit_into(img: Image.Image, cell: int, pad_color: Tuple[int, int, int]) -> Image.Image:
    """Scale to fit a cell x cell box preserving aspect, centred on padding."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (cell, cell), pad_color)
    scale = min(cell / w, cell / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    tile = Image.new("RGB", (cell, cell), pad_color)
    tile.paste(resized, ((cell - new_w) // 2, (cell - new_h) // 2))
    return tile


def make_single_image(img: Image.Image, size: int = 224) -> Image.Image:
    """Fit a single scene into a square canvas without distortion."""
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    tile = _fit_into(img.convert("RGB"), size, PAD_COLOR)
    canvas.paste(tile, (0, 0))
    return canvas


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_vqa_prompt(question: str) -> str:
    """Single-image visual question answering."""
    return f"answer en {question.strip()}"


def build_caption_prompt() -> str:
    """Single-image scene description. PaliGemma's native captioning prefix."""
    return "caption en"


def build_change_prompt(question: str) -> str:
    """Bi-temporal change VQA over a horizontal T1|T2 composite."""
    return f"answer en {CHANGE_PREFIX}{question.strip()}"


def build_fusion_prompt(question: str) -> str:
    """Cross-modal optical+SAR analysis over a horizontal optical|SAR composite."""
    return f"answer en {FUSION_PREFIX}{question.strip()}"


PROMPT_BUILDERS: Dict[str, object] = {
    "vqa": build_vqa_prompt,
    "caption": build_caption_prompt,
    "change": build_change_prompt,
    "fusion": build_fusion_prompt,
}


__all__ = [
    "CHANGE_PREFIX",
    "FUSION_PREFIX",
    "PAD_COLOR",
    "make_pair_composite",
    "make_single_image",
    "build_vqa_prompt",
    "build_caption_prompt",
    "build_change_prompt",
    "build_fusion_prompt",
    "PROMPT_BUILDERS",
]
