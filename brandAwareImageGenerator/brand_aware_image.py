# python -m venv .venv && source .venv/bin/activate
# pip install pillow numpy openai pydantic

"""
Brand-aware image pipeline (single pass)

Flow:
1) Prompt engineer -> produce single-line FINAL PROMPT using brand rules
2) Image generator -> OpenAI Images API or local Pillow stub
3) Brand check -> palette approximation and forbidden-terms validation
4) Content credentials -> write sidecar JSON next to the image

This file is standalone and safe to run without any external services.
"""

# ---- imports ----
import os
import json
import hashlib
import base64
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    raise ImportError("Please install the required Packages: pip install pydantic")

# Load environment variables from a local .env file if present (e.g., OPENAI_API_KEY).
from dotenv import load_dotenv

load_dotenv()

# If OpenAI SDK import fails, we silently fall back to the Pillow stub.
USE_OPENAI = False
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
    USE_OPENAI = False

# ---- Config ----
OUTPUT_DIR = "output"  # Output folders
BRAND_JSON_PATH = "brand.json"
DEFAULT_SIZE = "1024x1024"  # change DEFAULT_SIZE to e.g., '768x1024' for portrait.


class BrandModel(BaseModel):
    name: str = Field(..., description="Display name for the brand")
    palette_hex: List[str] = Field(..., description="Primary brand colors in hex, like ['#E50914', '#FFFFFF']")
    tone: str = Field(..., description="Short description of brand voice")
    required_terms: List[str] = Field(default_factory=list, description="Terms that should appear in copy if applicable")
    forbidden_terms: List[str] = Field(default_factory=list, description="Terms that must not appear in any generated text")
    logo_policy: Dict[str, Any] = Field(default_factory=dict, description="Placeholder for logo usage rules")
    max_colors: int = Field(default=4, description="Max colors preferred in hero compositions")


@dataclass
class GenerationResult:
    prompt: str
    base_image_path: str
    final_image_path: str
    credentials_path: str
    brand_check_report: Dict[str, Any]


# ----Helper functions----
def ensure_out_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # creates 'output' if missing; does nothing if it already exists


# To get the current TimeStamp
def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y5m5d_%H%M%S")


def parse_size(size_str: str) -> Tuple[int, int]:
    w, h = size_str.lower().split("x")
    return int(w), int(h)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in [0, 2, 4])  # type: ignore


def best_font() -> Optional[ImageFont.FreeTypeFont]:
    # Tries to load a common font for stubs; falls back to default if not found.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # for mac
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # linux
        "C:\\Windows\\Fonts\\arial.ttf",  # windows
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=20)
            except Exception:
                pass
    return None


# Example brand config written to brand.json if it doesn't exist.
SAMPLE_BRAND = {
    "name": "Red & White Skill Education",
    "palette_hex": ["#E50914", "#FFFFFF", "#1F1F1F"],
    "tone": "Friendly, instructional, trustworthy",
    "required_terms": ["Learn", "Career"],
    "forbidden_terms": ["Free money", "Guaranteed job"],
    "logo_policy": {"positions": ["top-right", "top-center", "bottom-right"], "min_clear_space": "12px"},
    "max_colors": 4,
}


def ensure_brand_json(path: str = BRAND_JSON_PATH) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_BRAND, f, indent=2)


def load_brand(path: str = BRAND_JSON_PATH) -> BrandModel:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        # Pydantic validates and converts raw dict 'data' into a typed BrandModel (raises ValidationError on mismatch).
        return BrandModel(**data)
    except ValidationError as e:
        raise SystemExit(f"Invalid brand json: {e}")


# ---- Prompt Engineering ----
def prompt_engineer(idea: str, brand: BrandModel) -> str:
    """
    Create a one line brand aware prompt.

    Rules:
        - include palette hints
        - keep tone cues
        - avoid generic or forbidden terms
    """

    colors = ", ".join(brand.palette_hex[: brand.max_colors])
    tone = brand.tone
    clean_idea = idea.strip().replace("\n", " ")
    prompt = f"FINAL PROMPT: {clean_idea}. Style: {tone}. palette: {colors}. Minimal clutter, high readability."
    return prompt


def generate_image_openai(prompt: str, size: str = DEFAULT_SIZE):
    if OpenAI is None:
        raise SystemError("OpenAi SDK is not available. Install openai and set the OPENAI_API_KEY")
    client = OpenAI()
    result = client.images.generate(model="gpt-image-1", prompt=prompt, size=size, n=1)
    b64 = result.data[0].b64_json

    from io import BytesIO

    # BytesIO use to wraps bytes as a file-like object so that PIL can open it without store to disk.
    return Image.open(BytesIO(base64.b64decode(b64)))


def generate_image_stub(prompt: str, brand: BrandModel, size: str = DEFAULT_SIZE):
    """
    Pillow-based placeholder image to simulate generated asset.
    Uses the first palette color as background and draws prompt text.
    """
    w, h = parse_size(size)
    background_color = hex_to_rgb(brand.palette_hex[0] if brand.palette_hex else "#DDDDDD")
    fontColor = (0, 0, 0) if np.mean(background_color) > 128 else (255, 255, 255)
    img = Image.new("RGB", (w, h), background_color)
    draw = ImageDraw.Draw(img)
    font = best_font()
    text = prompt[:160]
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(20, (w - text_width) // 2)
    y = max(20, (h - text_height) // 2)
    draw.text((x, y), text=text, fill=fontColor, font=font)
    return img


def write_png(img: Image.Image, prefix: str):
    """
    Store the Image to the output folder with the timestamp.
    """
    ensure_out_dir()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{ts()}.png")  # Combine prefix and timestamp with extension ".png" and then store the image
    img.save(path, format="PNG")
    return path


def check_forbidden_terms(prompt: str, forbidden: List[str]) -> List[str]:
    """
    Check the forbidden text in the prompt.
    """
    found = []
    prompt_lower = prompt.lower()
    for term in forbidden:
        if term.lower() in prompt_lower:
            found.append(term)
    return found


def dominant_colors(image: Image.Image, k: int = 5) -> List[Tuple[int, int, int]]:
    """
    Very simple K-means-like color extraction using downsample + unique counts.
    Good enough for a quick palette proximity check.
    """
    small = image.copy().resize((64, 64))  # Resize and check the color inside that part
    arr = np.array(small).reshape(-1, 3)

    # Count unique colors
    colors, counts = np.unique(arr, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    colors_sorted = [tuple(map(int, colors[i])) for i in order[:k]]
    return colors_sorted


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    Find the distance in RGB between the two colors.
    """
    return float(np.linalg.norm(np.array(c1) - np.array(c2)))  # linalg.norm computes Euclidean distance in RGB space


def palette_proximity_report(image: Image.Image, brand: BrandModel, threshold: float = 80.0) -> Dict[str, Any]:
    """
    Report whether dominant colors are reasonably close to brand palette.
    threshold is Euclidean distance in RGB space; lower is closer.
    """
    dom = dominant_colors(image, k=5)
    palette = [hex_to_rgb(hx) for hx in brand.palette_hex]
    distances = []
    for d in dom:
        closest = min(color_distance(d, p) for p in palette) if palette else 0.0
        distances.append({"dominant_color": d, "closest_distance": round(closest, 2)})
    ok = all(item["closest_distance"] <= threshold for item in distances) if palette else True
    return {"ok": ok, "threshold": threshold, "dominant_to_palette_distances": distances}


def run_brand_checks(prompt: str, image: Image.Image, brand: BrandModel) -> Dict[str, Any]:
    violations = check_forbidden_terms(prompt, brand.forbidden_terms)
    proximity = palette_proximity_report(image, brand)

    return {
        "forbidden_terms_found": violations,
        "palette_compliance": proximity["ok"],
        "palette_report": proximity,
        "required_terms_present": [term for term in brand.required_terms if term.lower() in prompt.lower()],
    }


def write_credentials_sidecar(
    image_path: str,
    prompt: str,
    brand: BrandModel,
    brand_check_report: Dict[str, Any],
    generator: str,
) -> str:
    """
    Writes a JSON sidecar acting as a placeholder for content credentials.
    Includes asset hash, brand data version hash, and generation notes.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    asset_hash = hashlib.sha256(img_bytes).hexdigest()
    brand_hash = hashlib.sha256(json.dumps(brand.model_dump(), sort_keys=True).encode("utf-8")).hexdigest()

    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "generator": generator,
        "prompt": prompt,
        "brand": {"name": brand.name, "hash": brand_hash},
        "asset": {"path": image_path, "sha256": asset_hash},
        "brand_check": brand_check_report,
        "provenance": {"c2pa_placeholder": True},
        "notes": "This file simulates Content Credentials metadata for interview practice.",
    }
    ensure_out_dir()
    cred_path = os.path.join(OUTPUT_DIR, f"credentials_{ts()}.json")
    with open(cred_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return cred_path


def run_pipeline(idea: str, size: str = DEFAULT_SIZE, use_openai: bool = False) -> GenerationResult:
    ensure_brand_json()
    brand = load_brand()

    prompt = prompt_engineer(idea=idea, brand=brand)

    if use_openai:
        if OpenAI is None or not os.environ.get("OPENAI_API_KEY"):
            raise SystemError("OpenAI mode required but SDK or OPENAI_API_KEY is missing.")
        img = generate_image_openai(prompt, size)
        generator_name = "openai:gpt-image-1"
    else:
        img = generate_image_stub(prompt, brand, size)
        generator_name = "stub:pillow"

    base_path = write_png(img, prefix="base")
    final_path = write_png(img, prefix="final")

    report = run_brand_checks(prompt, img, brand)

    cred_path = write_credentials_sidecar(final_path, prompt, brand, report, generator=generator_name)

    return GenerationResult(
        prompt=prompt,
        base_image_path=base_path,
        final_image_path=final_path,
        credentials_path=cred_path,
        brand_check_report=report,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brand Aware Image Generation Pipeline (Exercise 1)")
    parser.add_argument("--idea", required=True, help="Short idea for the image")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="WxH, e.g. 1024x1024")
    parser.add_argument("--use-openai", action="store_true", help="use OpenAI API instead of stub")  # 'action="store_true"' makes this a boolean flag: present -> True, absent -> False
    args = parser.parse_args()

    USE_OPENAI = bool(args.use_openai)

    result = run_pipeline(idea=args.idea, size=args.size, use_openai=USE_OPENAI)

    print("\n=== Generation Summary ===")
    print("Final prompt:", result.prompt)
    print("Base image:", result.base_image_path)
    print("Final image:", result.final_image_path)
    print("Credentials:", result.credentials_path)
    # print("Brand check report:", json.dumps(result.brand_check_report, indent=2))
