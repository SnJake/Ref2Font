import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

try:
    from skimage import measure
except ImportError:
    raise SystemExit("Please install scikit-image: pip install scikit-image")


CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"


def detect_background_is_white(img: Image.Image, border: int = 4) -> bool:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    border = max(1, int(border))
    top = gray[:border, :].ravel()
    bottom = gray[-border:, :].ravel()
    left = gray[:, :border].ravel()
    right = gray[:, -border:].ravel()
    if top.size + bottom.size + left.size + right.size == 0:
        return gray.mean() > 0.5
    border_vals = np.concatenate([top, bottom, left, right])
    return border_vals.mean() > 0.5


def baseline_from_cell(
    cell_img: Image.Image,
    invert: bool,
    threshold: int,
    quantile: float,
    min_pixels: int,
) -> float | None:
    arr = np.array(cell_img.convert("L"))
    if invert:
        mask = arr < threshold
    else:
        mask = arr > threshold
    ys = np.where(mask)[0]
    if ys.size < min_pixels:
        return None
    return float(np.quantile(ys, quantile))


def compute_grid(count: int, width: int, height: int) -> tuple[int, int, int, int, int]:
    best = None
    best_grid = (1, count, min(width, height) // count if count else min(width, height), 0, 0)
    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        cell = min(width // cols, height // rows)
        if cell <= 0:
            continue
        used_w = cell * cols
        used_h = cell * rows
        unused = (width - used_w) + (height - used_h)
        aspect_diff = abs((cols / rows) - (width / height))
        candidate = (cell, -unused, -aspect_diff)
        if best is None or candidate > best:
            best = candidate
            offset_x = (width - used_w) // 2
            offset_y = (height - used_h) // 2
            best_grid = (cols, rows, cell, offset_x, offset_y)
    return best_grid


def chaikin_smooth(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    if len(points) < 3:
        return points
    for _ in range(iterations):
        new_points = []
        num_pts = len(points)
        for i in range(num_pts):
            p0 = points[i]
            p1 = points[(i + 1) % num_pts]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new_points.append(Q)
            new_points.append(R)
        points = np.array(new_points)
    return points


def contours_from_image(
    img: Image.Image,
    level: float,
    simplify: float,
    trace_scale: int,
    trace_blur: float,
    smooth_iters: int,
    invert: bool,
) -> tuple[list[np.ndarray], tuple[float, float, float, float] | None]:
    
    # Подготовка
    work = img.convert("L")
    
    # Upscale для более плавного контура
    if trace_scale > 1:
        work = work.resize((img.width * trace_scale, img.height * trace_scale), resample=Image.Resampling.LANCZOS)
    
    if trace_blur > 0:
        work = work.filter(ImageFilter.GaussianBlur(radius=trace_blur))

    arr = np.array(work, dtype=np.float32) / 255.0
    
    if invert:
        arr = 1.0 - arr

    contours = measure.find_contours(arr, level=level)

    simplified = []
    all_points = []

    for contour in contours:
        if contour.shape[0] < 3:
            continue
        # skimage возвращает (row, col) -> (y, x). Нам нужно (x, y)
        pts = np.stack([contour[:, 1], contour[:, 0]], axis=1)
        

        if simplify > 0:
            pts = measure.approximate_polygon(pts, tolerance=simplify)
        

        if smooth_iters > 0 and pts.shape[0] > 3:
            pts = chaikin_smooth(pts, iterations=smooth_iters)


        if trace_scale > 1:
            pts = pts / float(trace_scale)
            
        if pts.shape[0] >= 3:
            simplified.append(pts)
            all_points.append(pts)
            
    bbox = None
    if all_points:
        all_pts_concat = np.vstack(all_points)
        min_x, min_y = all_pts_concat.min(axis=0)
        max_x, max_y = all_pts_concat.max(axis=0)
        bbox = (min_x, min_y, max_x, max_y)

    return simplified, bbox


def draw_glyph_fixed_grid(
    pen: TTGlyphPen, 
    contours: list[np.ndarray], 
    cell_bbox: tuple[int, int, int, int] | None, # Опционально для ширины
    scale: float, 
    side_bearing: int,
    cell_height: int,
    pixel_baseline: float,
    cell_width_ref: int
) -> int:

    

    if not contours:
        return int(cell_width_ref * scale * 0.5)

    g_min_x = float("inf")
    g_max_x = float("-inf")
    for contour in contours:
        xs = contour[:, 0]
        g_min_x = min(g_min_x, xs.min())
        g_max_x = max(g_max_x, xs.max())

    for contour in contours:
        # contour points are in cell-local coordinates (0..cell_w, 0..cell_h)
        # Convert to TTF coords (Y up). 0 is the baseline.
        # Formula: (pixel_baseline - y) * scale
        start_pt = contour[0]
        sx = (start_pt[0] - g_min_x) * scale + side_bearing
        sy = (pixel_baseline - start_pt[1]) * scale
        pen.moveTo((sx, sy))

        for pt in contour[1:]:
            px = (pt[0] - g_min_x) * scale + side_bearing
            py = (pixel_baseline - pt[1]) * scale
            pen.lineTo((px, py))
        pen.closePath()
    
    content_width = max(0, g_max_x - g_min_x)
    
    return int(content_width * scale + 2 * side_bearing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flux Grid to TTF (Fixed Alignment)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font-name", default="FluxFont")
    parser.add_argument("--charset", default=CHARSET)
    parser.add_argument("--canvas", type=int, default=2048)
    
    # Grid override
    parser.add_argument("--cols", type=int, default=None, help="Force number of columns")
    parser.add_argument("--rows", type=int, default=None, help="Force number of rows")

    # Metrics
    parser.add_argument("--upm", type=int, default=1024)
    parser.add_argument("--side-bearing", type=int, default=40)
    parser.add_argument("--baseline-ratio", type=float, default=0.75, help="Where is the baseline in the cell (0.0 top, 1.0 bottom)")
    parser.add_argument("--baseline-mode", choices=["fixed", "auto"], default="fixed", help="Baseline mode for grid alignment.")
    parser.add_argument("--baseline-quantile", type=float, default=0.9, help="Quantile for auto baseline (0..1).")
    parser.add_argument("--baseline-min-pixels", type=int, default=20, help="Min pixels to trust auto baseline in a cell.")
    parser.add_argument("--padding", type=float, default=0.1, help="Visual padding scale logic")

    # Vectorization
    parser.add_argument("--threshold", type=int, default=127) # Used if manual binarization needed, else auto
    parser.add_argument("--contour-level", type=float, default=0.5)
    parser.add_argument("--simplify", type=float, default=1.0)
    parser.add_argument("--trace-scale", type=int, default=4)
    parser.add_argument("--trace-blur", type=float, default=1.0)
    parser.add_argument("--smooth-iters", type=int, default=2)
    parser.add_argument("--invert", action="store_true", help="Force inversion before tracing.")
    parser.add_argument("--no-auto-invert", action="store_true", help="Disable auto inversion from background.")
    
    # Legacy args ignored but kept for compatibility with pipeline
    parser.add_argument("--vectorize", default="contours")
    parser.add_argument("--fixed-metrics", action="store_true")

    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("L")
    bg_is_white = detect_background_is_white(img)
    invert = args.invert or (not args.no_auto_invert and bg_is_white)
    
    # 1. Определяем сетку
    if args.cols and args.rows:
        cols, rows = args.cols, args.rows
        cell_size = min(img.width // cols, img.height // rows)
        cell_w = cell_size
        cell_h = cell_size
        offset_x = (img.width - cell_size * cols) // 2
        offset_y = (img.height - cell_size * rows) // 2
    else:
        # Auto-detect
        cols, rows, cell_size_sq, offset_x, offset_y = compute_grid(len(args.charset), img.width, img.height)
        cell_w = cell_size_sq
        cell_h = cell_size_sq

    print(f"Grid: {cols}x{rows}, Cell: {cell_w}x{cell_h}")

    row_baselines: dict[int, float] = {}
    baseline_ratio_used = args.baseline_ratio
    if args.baseline_mode == "auto":
        per_row: dict[int, list[float]] = {r: [] for r in range(rows)}
        for idx, _ch in enumerate(args.charset):
            row = idx // cols
            col = idx % cols
            x0 = offset_x + col * cell_w
            y0 = offset_y + row * cell_h
            cell_img = img.crop((x0, y0, x0 + cell_w, y0 + cell_h))
            baseline = baseline_from_cell(
                cell_img, invert, args.threshold, args.baseline_quantile, args.baseline_min_pixels
            )
            if baseline is not None:
                per_row[row].append(baseline)
        for row, vals in per_row.items():
            if vals:
                row_baselines[row] = float(np.median(vals))
        if row_baselines:
            baseline_ratio_used = float(np.median(list(row_baselines.values()))) / float(cell_h)

    scale = args.upm / float(cell_h) * (1.0 - args.padding) 

    ascent = int(args.upm * baseline_ratio_used)
    descent = int(args.upm * (baseline_ratio_used - 1.0))

    
    fb = FontBuilder(args.upm, isTTF=True)
    glyph_order = [".notdef"] + list(args.charset)
    fb.setupGlyphOrder(glyph_order)
    fb.setupNameTable(dict(familyName=args.font_name, styleName="Regular", uniqueFontIdentifier=args.font_name, fullName=args.font_name, psName=args.font_name.replace(" ", "-"), version="Version 1.0"))
    fb.setupPost()

    glyph_dict = {}
    metrics = {}

    # .notdef glyph
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0)); pen.lineTo((50, args.upm)); pen.lineTo((450, args.upm)); pen.lineTo((450, 0)); pen.closePath()
    glyph_dict[".notdef"] = pen.glyph()
    metrics[".notdef"] = (500, 50)

    for idx, ch in enumerate(args.charset):
        row = idx // cols
        col = idx % cols
        
        # Координаты ячейки
        x0 = offset_x + col * cell_w
        y0 = offset_y + row * cell_h
        
        # Crop
        cell_img = img.crop((x0, y0, x0 + cell_w, y0 + cell_h))
        
        contours, _ = contours_from_image(
            cell_img,
            level=args.contour_level,
            simplify=args.simplify,
            trace_scale=args.trace_scale,
            trace_blur=args.trace_blur,
            smooth_iters=args.smooth_iters,
            invert=invert,
        )
        
        pen = TTGlyphPen(None)
        
        advance = draw_glyph_fixed_grid(
            pen, 
            contours, 
            None, 
            scale, 
            args.side_bearing, 
            cell_h,
            row_baselines.get(row, cell_h * baseline_ratio_used),
            cell_w
        )

        glyph_dict[ch] = pen.glyph()
        metrics[ch] = (advance, args.side_bearing)

    fb.setupGlyf(glyph_dict)
    fb.setupCharacterMap({ord(c): c for c in args.charset})
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupOS2(sTypoAscender=ascent, usWinAscent=ascent, usWinDescent=abs(descent))

    out_path = output_dir / f"{args.font_name.replace(' ', '_')}.ttf"
    fb.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

