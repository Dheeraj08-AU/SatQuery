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

import re
from typing import Dict, List, Literal, Optional, Sequence, Tuple

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


def fit_transform(
    orig_size: Tuple[int, int], size: int
) -> Tuple[float, int, int, int, int]:
    """
    Geometry of `_fit_into`: how an original image maps into a square canvas.

    Returns (scale, dx, dy, new_w, new_h). A point (x, y) in the original image
    lands at (x * scale + dx, y * scale + dy) on the canvas.

    Needed because grounding boxes are annotated in the ORIGINAL image's pixel
    space, but the model sees the letterboxed canvas. Training on untransformed
    coordinates teaches the model boxes that do not correspond to what it is
    looking at.
    """
    w, h = orig_size
    if w <= 0 or h <= 0:
        return 1.0, 0, 0, size, size
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    dx = (size - new_w) // 2
    dy = (size - new_h) // 2
    return scale, dx, dy, new_w, new_h


def box_to_canvas(
    box: Sequence[float], orig_size: Tuple[int, int], size: int = 224
) -> List[float]:
    """Map [x0, y0, x1, y1] from original image pixels into canvas pixels."""
    scale, dx, dy, _, _ = fit_transform(orig_size, size)
    x0, y0, x1, y1 = box
    return [x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy]


def box_from_canvas(
    box: Sequence[float], orig_size: Tuple[int, int], size: int = 224
) -> List[float]:
    """Inverse of `box_to_canvas`: canvas pixels back to original pixels."""
    scale, dx, dy, _, _ = fit_transform(orig_size, size)
    if scale <= 0:
        return list(box)
    x0, y0, x1, y1 = box
    w, h = orig_size
    out = [(x0 - dx) / scale, (y0 - dy) / scale, (x1 - dx) / scale, (y1 - dy) / scale]
    return [
        max(0.0, min(out[0], w)),
        max(0.0, min(out[1], h)),
        max(0.0, min(out[2], w)),
        max(0.0, min(out[3], h)),
    ]


# ---------------------------------------------------------------------------
# PaliGemma location tokens
# ---------------------------------------------------------------------------
#
# PaliGemma has 1024 reserved location tokens, <loc0000> to <loc1023>, and was
# pretrained on a `detect {label}` task that emits them. Fine-tuning that native
# capability on VRSBench referring expressions gives a remote-sensing-adapted
# grounding component using the exact training pipeline already built for VQA -
# far cheaper on a T4 than fine-tuning a separate detector, and it satisfies the
# problem statement's requirement that specialist components be domain-adapted.
#
# Token order is y_min, x_min, y_max, x_max, each normalised to [0, 1023] over
# the image handed to the processor.

LOC_BINS = 1024
_LOC_RE = re.compile(r"<loc(\d{4})>")


def encode_loc_tokens(box: Sequence[float], canvas_size: int = 224) -> str:
    """[x0, y0, x1, y1] in canvas pixels -> '<locYYYY><locXXXX><locYYYY><locXXXX>'."""
    x0, y0, x1, y1 = box
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    def q(v: float) -> int:
        frac = v / float(canvas_size)
        return max(0, min(LOC_BINS - 1, int(round(frac * (LOC_BINS - 1)))))

    return f"<loc{q(y0):04d}><loc{q(x0):04d}><loc{q(y1):04d}><loc{q(x1):04d}>"


def decode_loc_tokens(
    text: str, canvas_size: int = 224
) -> List[Tuple[List[float], str]]:
    """
    Parse a PaliGemma detection string into [(box_xyxy_canvas, label), ...].

    Handles multiple detections separated by ' ; ', and tolerates a missing or
    empty label. Groups of four location tokens are consumed in order; any
    trailing text before the next group is taken as that box's label.
    """
    out: List[Tuple[List[float], str]] = []
    for chunk in text.split(";"):
        tokens = _LOC_RE.findall(chunk)
        if len(tokens) < 4:
            continue
        y0, x0, y1, x1 = (int(t) for t in tokens[:4])
        scale = canvas_size / float(LOC_BINS - 1)
        box = [x0 * scale, y0 * scale, x1 * scale, y1 * scale]
        label = _LOC_RE.sub("", chunk).strip() or "region"
        out.append((box, label))
    return out


def build_detect_prompt(phrases: Sequence[str]) -> str:
    """PaliGemma's native detection prefix. Multiple targets are ';'-separated."""
    cleaned = [p.strip().rstrip(".").strip() for p in phrases if str(p).strip()]
    if not cleaned:
        cleaned = ["object"]
    return "detect " + " ; ".join(cleaned)


def build_detect_suffix(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    canvas_size: int = 224,
) -> str:
    """Training target for the detection task."""
    parts = []
    for box, label in zip(boxes, labels):
        parts.append(f"{encode_loc_tokens(box, canvas_size)} {label.strip().rstrip('.')}")
    return " ; ".join(parts)


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
    "LOC_BINS",
    "fit_transform",
    "box_to_canvas",
    "box_from_canvas",
    "encode_loc_tokens",
    "decode_loc_tokens",
    "build_detect_prompt",
    "build_detect_suffix",
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
