import argparse
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

import flux_upscale
import numpy as np
from PIL import Image


CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = value.strip("_")
    return value or "font"


def build_atlas_cmd(
    atlas_script: Path,
    image_path: Path,
    output_dir: Path,
    font_name: str,
    args,
    debug_dir: Path | None,
    use_grid: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(atlas_script),
        "--image",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--font-name",
        font_name,
        "--charset",
        args.charset,
        "--threshold",
        str(args.threshold),
        "--upm",
        str(args.upm),
        "--padding",
        str(args.padding),
        "--side-bearing",
        str(args.side_bearing),
    ]
    if use_grid:
        cmd.extend(["--canvas", str(args.canvas)])
        if args.cols:
            cmd.extend(["--cols", str(args.cols)])
        if args.rows:
            cmd.extend(["--rows", str(args.rows)])
            
        cmd.extend(["--simplify", str(args.simplify)])
        cmd.extend(["--baseline-ratio", str(args.baseline_ratio)])
        cmd.extend(["--baseline-mode", str(args.baseline_mode)])
        cmd.extend(["--baseline-quantile", str(args.baseline_quantile)])
        cmd.extend(["--baseline-min-pixels", str(args.baseline_min_pixels)])
        cmd.extend(["--contour-level", str(args.contour_level)])
        cmd.extend(["--trace-scale", str(args.trace_scale)])
        cmd.extend(["--trace-blur", str(args.trace_blur)])
        cmd.extend(["--smooth-iters", str(args.smooth_iters)])
        cmd.extend(["--descender-chars", str(args.descender_chars)])
        cmd.extend(["--descender-lift", str(args.descender_lift)])
        if not args.no_clean_components:
            cmd.append("--clean-components")
            cmd.extend(["--keep-components", str(args.keep_components)])
            cmd.extend(["--min-component-area", str(args.min_component_area)])
            cmd.extend(["--component-center-bias", str(args.component_center_bias)])
        cmd.extend(["--cell-bleed", str(args.cell_bleed)])
        cmd.extend(["--cell-bleed-max", str(args.cell_bleed_max)])
        if args.glyph_post_scale is not None:
            cmd.extend(["--glyph-post-scale", str(args.glyph_post_scale)])
        if args.invert:
            cmd.append("--invert")
        if args.no_auto_invert:
            cmd.append("--no-auto-invert")
    else:
        cmd.extend(
            [
                "--row-gap",
                str(args.row_gap),
                "--gap-threshold",
                str(args.gap_threshold),
                "--min-width-ratio",
                str(args.min_width_ratio),
                "--downsample",
                str(args.downsample),
            ]
        )
    if debug_dir is not None and not use_grid:
        cmd.extend(["--debug-dir", str(debug_dir)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Upscale atlas images and convert to TTF.")
    parser.add_argument("--input", required=True, help="Input atlas image or directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for TTF fonts.")
    parser.add_argument("--no-upscale", action="store_true", help="Skip upscaling.")
    parser.add_argument("--upscaled-dir", default=None)
    parser.add_argument("--model", default=r"G:\Programs\FontNN\upscaler_best_1E.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--target-scale", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-overlap", type=int, default=0)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--pad-mode", default="reflect")
    parser.add_argument("--format", choices=["png", "webp"], default="png")

    parser.add_argument("--charset", default=CHARSET)
    parser.add_argument("--threshold", type=int, default=127)
    
    # Grid override options
    parser.add_argument("--cols", type=int, default=None, help="Force grid columns")
    parser.add_argument("--rows", type=int, default=None, help="Force grid rows")

    parser.add_argument("--row-gap", type=int, default=2)
    parser.add_argument("--gap-threshold", type=int, default=0)
    parser.add_argument("--min-width-ratio", type=float, default=0.2)
    parser.add_argument("--downsample", type=float, default=4.0)
    parser.add_argument("--upm", type=int, default=1024)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--side-bearing", type=int, default=50)
    parser.add_argument("--debug-dir", default=None)

    parser.add_argument("--use-grid", action="store_true", help="Use grid-based atlas to TTF conversion.")
    parser.add_argument("--canvas", type=int, default=2048, help="Atlas canvas size for grid conversion.")
    parser.add_argument("--post-threshold", type=int, default=None)
    parser.add_argument(
        "--no-clean-components",
        action="store_true",
        help="Disable per-cell connected-component cleanup in grid conversion.",
    )
    parser.add_argument(
        "--keep-components",
        type=int,
        default=3,
        help="How many connected components to keep per cell (grid cleanup).",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=6,
        help="Minimum connected-component area in pixels (grid cleanup).",
    )
    parser.add_argument(
        "--component-center-bias",
        type=float,
        default=0.25,
        help="Center preference for component ranking in [0,1] (grid cleanup).",
    )
    parser.add_argument(
        "--cell-bleed",
        type=float,
        default=0.08,
        help="Extra margin around each grid cell for contour extraction.",
    )
    parser.add_argument(
        "--cell-bleed-max",
        type=int,
        default=24,
        help="Maximum bleed in pixels around each grid cell.",
    )

    parser.add_argument("--vectorize", choices=["rects", "contours"], default="contours")
    parser.add_argument("--simplify", type=float, default=0.8)
    parser.add_argument("--fixed-metrics", action="store_true")
    parser.add_argument("--baseline-ratio", type=float, default=0.75, help="Baseline position (0=top, 1=bottom)")
    parser.add_argument("--baseline-mode", choices=["fixed", "auto"], default="fixed", help="Baseline mode for grid alignment.")
    parser.add_argument("--baseline-quantile", type=float, default=0.9, help="Quantile for auto baseline (0..1).")
    parser.add_argument("--baseline-min-pixels", type=int, default=20, help="Min pixels to trust auto baseline in a cell.")
    parser.add_argument("--contour-level", type=float, default=0.5)
    parser.add_argument("--trace-scale", type=int, default=8)
    parser.add_argument("--trace-blur", type=float, default=1.0)
    parser.add_argument("--smooth-iters", type=int, default=2)
    parser.add_argument(
        "--descender-chars",
        default="gjpqy",
        help="Characters to lift up slightly in grid conversion.",
    )
    parser.add_argument(
        "--descender-lift",
        type=float,
        default=0.02,
        help="Lift amount for descender chars (fraction of cell height).",
    )
    parser.add_argument(
        "--glyph-post-scale",
        type=float,
        default=None,
        help="Optional final glyph scale multiplier before TTF save (e.g. 1.1).",
    )
    parser.add_argument("--invert", action="store_true", help="Force inversion before tracing (grid).")
    parser.add_argument("--no-auto-invert", action="store_true", help="Disable auto inversion (grid).")

    args = parser.parse_args()
    if args.glyph_post_scale is not None and args.glyph_post_scale <= 0:
        raise SystemExit("glyph-post-scale must be > 0")
    if args.keep_components < 1:
        raise SystemExit("--keep-components must be >= 1.")
    if args.min_component_area < 1:
        raise SystemExit("--min-component-area must be >= 1.")
    if args.component_center_bias < 0 or args.component_center_bias > 1:
        raise SystemExit("--component-center-bias must be in [0, 1].")
    if args.cell_bleed < 0:
        raise SystemExit("--cell-bleed must be >= 0.")
    if args.cell_bleed_max < 0:
        raise SystemExit("--cell-bleed-max must be >= 0.")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    base_dir = Path(__file__).resolve().parent
    atlas_script = base_dir / "scripts" / "atlas_to_ttf.py"
    grid_script = base_dir / "flux_grid_to_ttf.py"
    
    if args.use_grid and not grid_script.exists():
         raise SystemExit(f"flux_grid_to_ttf.py not found: {grid_script}")

    upscaled_dir = Path(args.upscaled_dir) if args.upscaled_dir else output_dir / "upscaled"
    upscaled_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    images = flux_upscale.iter_images(input_path)
    if not images:
        raise SystemExit("No input images found")

    if args.no_upscale:
        upscaled_images = images
    else:
        model, _ = flux_upscale.load_checkpoint(Path(args.model))
        device = flux_upscale.choose_device(args.device)
        model = model.to(device)

        upscaled_images = []
        for path in tqdm(images, desc="Upscaling"):
            img = flux_upscale.load_image(path)
            out = flux_upscale.upscale_image(
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
            out_path = (upscaled_dir / rel).with_suffix(f".{args.format}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.post_threshold is not None:
                arr = np.array(out.convert("L"))
                thresh = args.post_threshold
                mean_val = float(arr.mean())
                if mean_val < 128:
                    mask = arr > thresh
                else:
                    mask = arr < thresh
                out = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            if args.format == "webp":
                out.save(out_path, format="WEBP", lossless=True, quality=100, method=6)
            else:
                out.save(out_path, format="PNG")
            upscaled_images.append(out_path)

    for image_path in tqdm(upscaled_images, desc="Converting to TTF"):
        font_name = sanitize_name(image_path.stem)
        font_debug = debug_dir / font_name if debug_dir else None
        script = grid_script if args.use_grid else atlas_script
        
        # build_atlas_cmd теперь содержит новые параметры
        cmd = build_atlas_cmd(
            script,
            image_path,
            output_dir,
            font_name,
            args,
            font_debug,
            args.use_grid,
        )
        
        subprocess.run(cmd, check=True)

    print(f"Done. Generated {len(upscaled_images)} font(s) in {output_dir}")


if __name__ == "__main__":
    main()
