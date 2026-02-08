from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from fontnn.models.atlas_upscaler import AtlasSuperResolutionNet, UpscalerConfig


Upscaler = AtlasSuperResolutionNet


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError("Checkpoint must contain one of: model, model_state_dict, state_dict")


def _extract_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw = checkpoint.get("config", {})
    if not isinstance(raw, dict):
        raw = {}

    cfg = {
        "in_channels": int(raw.get("in_channels", 1)),
        "out_channels": int(raw.get("out_channels", 1)),
        "feature_channels": int(raw.get("feature_channels", 96)),
        "num_residual_blocks": int(raw.get("num_residual_blocks", 16)),
        "residual_expansion": int(raw.get("residual_expansion", 4)),
        "dropout": float(raw.get("dropout", 0.0)),
        # Keep 4 as compatibility default for legacy checkpoints that omitted config.
        "upscale_factor": int(raw.get("upscale_factor", 4)),
    }
    return cfg


def load_checkpoint(checkpoint_path: Path) -> tuple[Upscaler, dict[str, Any]]:
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        from torch.serialization import safe_globals

        with safe_globals([pathlib.PosixPath]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")

    model_config = _extract_model_config(checkpoint)
    model = Upscaler(UpscalerConfig(**model_config))
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_config


def choose_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def from_tensor(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0.0, 1.0).squeeze(0).squeeze(0)
    arr = (t * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr)


def upscale_image(
    model: Upscaler,
    device: torch.device,
    img: Image.Image,
    passes: int,
    target_scale: int | None,
    tile_size: int | None = None,
    tile_overlap: int = 0,
    pad: int = 0,
    pad_mode: str = "reflect",
) -> Image.Image:
    if tile_size is not None and tile_size <= 0:
        tile_size = None
    if tile_overlap < 0:
        tile_overlap = 0
    if tile_size is not None and tile_overlap >= tile_size:
        tile_overlap = max(0, tile_size - 1)

    x = to_tensor(img).to(device)
    scale_per_pass = int(getattr(model, "upscale_factor", 1))
    total_scale = scale_per_pass**passes

    if tile_size is None or (x.shape[-1] <= tile_size and x.shape[-2] <= tile_size):
        with torch.no_grad():
            for _ in range(passes):
                x = model(x)
        out = from_tensor(x)
    else:
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
        h, w = x.shape[-2:]
        out_h = h * total_scale
        out_w = w * total_scale
        output = torch.zeros((1, 1, out_h, out_w), device=device)
        weight = torch.zeros((1, 1, out_h, out_w), device=device)

        step = max(1, tile_size - tile_overlap)
        for y in range(0, h, step):
            for x0 in range(0, w, step):
                tile = x[:, :, y : y + tile_size, x0 : x0 + tile_size]
                pad_bottom = max(0, tile_size - tile.shape[-2])
                pad_right = max(0, tile_size - tile.shape[-1])
                if pad_bottom or pad_right:
                    effective_mode = pad_mode
                    if pad_mode in {"reflect", "replicate"}:
                        if pad_right >= tile.shape[-1] or pad_bottom >= tile.shape[-2]:
                            effective_mode = "constant"
                    tile = F.pad(tile, (0, pad_right, 0, pad_bottom), mode=effective_mode)

                with torch.no_grad():
                    out_tile = tile
                    for _ in range(passes):
                        out_tile = model(out_tile)

                out_y = y * total_scale
                out_x = x0 * total_scale
                out_h_tile = tile_size * total_scale
                out_w_tile = tile_size * total_scale
                if pad_bottom or pad_right:
                    out_h_tile = (tile.shape[-2] - pad_bottom) * total_scale
                    out_w_tile = (tile.shape[-1] - pad_right) * total_scale
                output[:, :, out_y : out_y + out_h_tile, out_x : out_x + out_w_tile] += out_tile[
                    :, :, :out_h_tile, :out_w_tile
                ]
                weight[:, :, out_y : out_y + out_h_tile, out_x : out_x + out_w_tile] += 1.0

        output = output / torch.clamp(weight, min=1.0)
        if pad > 0:
            crop = pad * total_scale
            output = output[:, :, crop : out_h - crop, crop : out_w - crop]
        out = from_tensor(output)

    if target_scale is not None:
        desired_w = img.width * target_scale
        desired_h = img.height * target_scale
        if out.size != (desired_w, desired_h):
            out = out.resize((desired_w, desired_h), Image.Resampling.LANCZOS)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Upscale atlas images with a trained atlas upscaler checkpoint.")
    parser.add_argument("--input", required=True, help="Input image or directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for upscaled images.")
    parser.add_argument(
        "--model",
        default=r"G:\Programs\FontNN\upscaler_best_1E.pt",
        help="Path to upscaler checkpoint (.pt).",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="How many times to apply the model (scale multiplies per pass).",
    )
    parser.add_argument(
        "--target-scale",
        type=int,
        default=None,
        help="Final scale relative to input (e.g. 2 or 4). If set, output is resized to match.",
    )
    parser.add_argument("--tile-size", type=int, default=None, help="Process image in tiles to save VRAM.")
    parser.add_argument("--tile-overlap", type=int, default=0, help="Overlap between tiles in pixels.")
    parser.add_argument("--pad", type=int, default=0, help="Padding (in pixels) before tiling.")
    parser.add_argument(
        "--pad-mode",
        choices=["reflect", "replicate", "constant"],
        default="reflect",
        help="Padding mode for tiling.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "webp"],
        default="png",
        help="Output image format.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_config = load_checkpoint(Path(args.model))
    device = choose_device(args.device)
    model = model.to(device)

    if args.passes < 1:
        raise SystemExit("passes must be >= 1")

    images = iter_images(input_path)
    if not images:
        raise SystemExit("No input images found")

    for path in tqdm(images, desc="Upscaling"):
        img = load_image(path)
        out = upscale_image(
            model,
            device,
            img,
            passes=args.passes,
            target_scale=args.target_scale,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            pad=args.pad,
            pad_mode=args.pad_mode,
        )
        rel = path.name if input_path.is_file() else path.relative_to(input_path)
        out_path = (output_dir / rel).with_suffix(f".{args.format}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "webp":
            out.save(out_path, format="WEBP", lossless=True, quality=100, method=6)
        else:
            out.save(out_path, format="PNG")

    print(
        f"Done. Saved {len(images)} image(s) to {output_dir}. "
        f"Model scale-per-pass={model_config.get('upscale_factor', 1)}."
    )


if __name__ == "__main__":
    main()
