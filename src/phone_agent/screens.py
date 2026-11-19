"""Shrink screenshots to the size the model will see.

Muse Glimmer cuts an image into 28x28-pixel squares and turns each square into one
token. A full 1080x2400 screenshot would be about 3,300 tokens, and tokens are what we
pay for (and what slows training). So we shrink each screenshot to fit a token budget.

We resize to an exact multiple of 28 pixels on both sides. Then the model's own
processor has nothing left to resize, and we know the token count exactly. The
aspect ratio can shift by up to one square; labels don't care, because they're 0-1000
fractions of the whole picture, not pixels.
"""

from __future__ import annotations

import io
import math

from PIL import Image

PATCH_PX = 28  # Glimmer: patch 14 px x merge 2 = one token per 28x28 square
DEFAULT_MAX_IMAGE_TOKENS = 1000  # 1080x2400 -> 588x1316 = 21 x 47 = 987 tokens
JPEG_QUALITY = 90


def model_grid(width: int, height: int, max_tokens: int = DEFAULT_MAX_IMAGE_TOKENS) -> tuple[int, int]:
    """(columns, rows) of 28-px squares: as big as possible within max_tokens, never upscaled."""
    if width <= 0 or height <= 0 or max_tokens <= 0:
        raise ValueError(f"bad size {width}x{height} or budget {max_tokens}")
    full = (width / PATCH_PX) * (height / PATCH_PX)
    scale = min(1.0, math.sqrt(max_tokens / full))
    cols = max(1, math.floor(width * scale / PATCH_PX))
    rows = max(1, math.floor(height * scale / PATCH_PX))
    return cols, rows


def image_tokens(width: int, height: int, max_tokens: int = DEFAULT_MAX_IMAGE_TOKENS) -> int:
    cols, rows = model_grid(width, height, max_tokens)
    return cols * rows


def resize_for_model(png: bytes, max_tokens: int = DEFAULT_MAX_IMAGE_TOKENS) -> Image.Image:
    """Decode a screenshot and resize it to the model grid (RGB, sides multiple of 28)."""
    with Image.open(io.BytesIO(png)) as image:
        image = image.convert("RGB")
        cols, rows = model_grid(image.width, image.height, max_tokens)
        return image.resize((cols * PATCH_PX, rows * PATCH_PX), Image.Resampling.LANCZOS)


def to_jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
