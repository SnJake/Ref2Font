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
    
    # Инверсия (белые буквы на черном или наоборот). 
    # Предполагаем, что буквы светлее фона, если среднее < 0.5, иначе инвертируем.
    # Но для надежности лучше считать, что threshold отделяет объект.
    # Обычно find_contours ищет уровень. Если буквы белые (1.0), level 0.5 работает.
    # Если буквы черные (0.0), нужно инвертировать.
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
        
        # 1. Сначала упрощаем, чтобы убрать шум пикселей
        if simplify > 0:
            pts = measure.approximate_polygon(pts, tolerance=simplify)
        
        # 2. Сглаживаем углы (Chaikin), пока координаты еще крупные
        if smooth_iters > 0 and pts.shape[0] > 3:
            pts = chaikin_smooth(pts, iterations=smooth_iters)

        # 3. Возвращаем к оригинальному масштабу
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
    cell_width_ref: int,
    y_shift_px: float = 0.0,
) -> int:
    """
    Рисует глиф, полагаясь на координаты внутри ячейки, а не на bbox глифа.
    Это предотвращает "пляску" букв.
    """
    
    # Y-координата базовой линии в пикселях картинки (сверху вниз)
    # pixel_baseline is in cell-local pixel coords (Y down from top).

    if not contours:
        return int(cell_width_ref * scale * 0.5)

    # Вычисляем ширину глифа для Advance Width (отступ справа)
    # Но рисуем все равно относительно 0,0 ячейки
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
        sy = (pixel_baseline - (start_pt[1] - y_shift_px)) * scale
        pen.moveTo((sx, sy))

        for pt in contour[1:]:
            px = (pt[0] - g_min_x) * scale + side_bearing
            py = (pixel_baseline - (pt[1] - y_shift_px)) * scale
            pen.lineTo((px, py))
        pen.closePath()
    
    # Advance width: ширина контента + боковые отступы
    # Либо фиксированная ширина ячейки, если шрифт моноширинный
    content_width = max(0, g_max_x - g_min_x)
    
    # Если хотим "плотный" шрифт, берем ширину контента.
    # Если хотим, как в атласе (моноширинно), берем cell_width_ref.
    # Обычно для Flux атласов лучше брать контент + bearing, иначе пробелы огромные.
    return int(content_width * scale + 2 * side_bearing)

def contours_bounds(contours: list[np.ndarray]) -> tuple[float, float, float, float] | None:
    if not contours:
        return None
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for contour in contours:
        if contour.size == 0:
            continue
        xs = contour[:, 0]
        ys = contour[:, 1]
        min_x = min(min_x, float(xs.min()))
        min_y = min(min_y, float(ys.min()))
        max_x = max(max_x, float(xs.max()))
        max_y = max(max_y, float(ys.max()))
    if min_x == float("inf"):
        return None
    return (min_x, min_y, max_x, max_y)


def clean_cell_components(
    cell_img: Image.Image,
    invert: bool,
    threshold: int,
    keep_components: int,
    min_component_area: int,
    center_bias: float,
    core_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    arr = np.array(cell_img.convert("L"), dtype=np.uint8)
    bg_val = 255 if invert else 0
    if invert:
        fg = arr < threshold
    else:
        fg = arr > threshold

    if not np.any(fg):
        return Image.fromarray(np.full_like(arr, bg_val, dtype=np.uint8), mode="L")

    labels = measure.label(fg, connectivity=2)
    if labels.max() <= 0:
        return Image.fromarray(np.full_like(arr, bg_val, dtype=np.uint8), mode="L")

    h, w = fg.shape
    if core_box is None:
        core_box = (0, 0, w, h)
    core_x0, core_y0, core_x1, core_y1 = core_box
    core_x0 = max(0, min(w, int(core_x0)))
    core_y0 = max(0, min(h, int(core_y0)))
    core_x1 = max(core_x0, min(w, int(core_x1)))
    core_y1 = max(core_y0, min(h, int(core_y1)))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    diag = max(1.0, math.hypot(cx, cy))

    candidates_core: list[tuple[float, int]] = []
    candidates_other: list[tuple[float, int]] = []
    fallback: list[tuple[int, int]] = []
    for region in measure.regionprops(labels):
        area = int(region.area)
        fallback.append((area, int(region.label)))
        if area < int(min_component_area):
            continue
        minr, minc, maxr, maxc = region.bbox
        intersects_core = (maxc > core_x0) and (minc < core_x1) and (maxr > core_y0) and (minr < core_y1)
        ry, rx = region.centroid
        dist_norm = float(math.hypot(float(rx) - cx, float(ry) - cy) / diag)
        score = float(area) * (1.0 - float(center_bias) * dist_norm)
        if intersects_core:
            candidates_core.append((score, int(region.label)))
        else:
            candidates_other.append((score, int(region.label)))

    if candidates_core:
        candidates_core.sort(key=lambda x: x[0], reverse=True)
        keep_labels = {label for _, label in candidates_core[: max(1, int(keep_components))]}
    elif candidates_other:
        candidates_other.sort(key=lambda x: x[0], reverse=True)
        keep_labels = {label for _, label in candidates_other[: max(1, int(keep_components))]}
    else:
        fallback.sort(key=lambda x: x[0], reverse=True)
        keep_labels = {fallback[0][1]} if fallback else set()

    cleaned = np.isin(labels, list(keep_labels))
    out = np.full_like(arr, bg_val, dtype=np.uint8)
    out[cleaned] = arr[cleaned]
    return Image.fromarray(out, mode="L")


def component_limit_for_char(ch: str, max_keep: int) -> int:
    # Most glyphs are single connected component.
    # Keep 2 for common multi-part punctuation/letters.
    multi = {"i", "j", ":", ";", "!", "?"}
    target = 2 if ch in multi else 1
    return max(1, min(int(max_keep), target))


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
    parser.add_argument("--padding", type=float, default=0.1, help="Visual padding ratio in em units.")
    parser.add_argument(
        "--descender-chars",
        default="gjpqy",
        help="Characters to lift slightly to reduce descender clipping.",
    )
    parser.add_argument(
        "--descender-lift",
        type=float,
        default=0.02,
        help="Lift amount for descender chars as a fraction of cell height.",
    )
    parser.add_argument(
        "--glyph-post-scale",
        type=float,
        default=None,
        help="Optional final glyph scale multiplier before TTF save (e.g. 1.1).",
    )
    parser.add_argument(
        "--clean-components",
        action="store_true",
        help="Clean per-cell connected components to remove small neighbor leaks.",
    )
    parser.add_argument(
        "--keep-components",
        type=int,
        default=3,
        help="How many connected components to keep per cell (for clean-components).",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=6,
        help="Minimum connected-component area in pixels (for clean-components).",
    )
    parser.add_argument(
        "--component-center-bias",
        type=float,
        default=0.25,
        help="Center preference when ranking components, 0..1 (for clean-components).",
    )
    parser.add_argument(
        "--cell-bleed",
        type=float,
        default=0.08,
        help="Extra margin around each cell for contour extraction, as a fraction of cell size.",
    )
    parser.add_argument(
        "--cell-bleed-max",
        type=int,
        default=24,
        help="Maximum bleed in pixels around each cell.",
    )

    # Vectorization
    parser.add_argument("--threshold", type=int, default=127)
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

    if not (0 < args.baseline_ratio < 1):
        raise SystemExit("baseline-ratio must be between 0 and 1")
    if args.padding < 0 or args.padding >= 0.5:
        raise SystemExit("padding must be in [0, 0.5)")
    if args.descender_lift < 0:
        raise SystemExit("descender-lift must be >= 0")
    if args.glyph_post_scale is not None and args.glyph_post_scale <= 0:
        raise SystemExit("glyph-post-scale must be > 0")
    if args.keep_components < 1:
        raise SystemExit("keep-components must be >= 1")
    if args.min_component_area < 1:
        raise SystemExit("min-component-area must be >= 1")
    if args.component_center_bias < 0 or args.component_center_bias > 1:
        raise SystemExit("component-center-bias must be in [0, 1]")
    if args.cell_bleed < 0:
        raise SystemExit("cell-bleed must be >= 0")
    if args.cell_bleed_max < 0:
        raise SystemExit("cell-bleed-max must be >= 0")

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("L")
    bg_is_white = detect_background_is_white(img)
    invert = args.invert or (not args.no_auto_invert and bg_is_white)

    if args.cols and args.rows:
        cols, rows = args.cols, args.rows
        cell_size = min(img.width // cols, img.height // rows)
        cell_w = cell_size
        cell_h = cell_size
        offset_x = (img.width - cell_size * cols) // 2
        offset_y = (img.height - cell_size * rows) // 2
    else:
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

    descender_chars = set(args.descender_chars or "")
    glyph_entries: list[dict] = []
    max_above = 0.0
    max_below = 0.0

    for idx, ch in enumerate(args.charset):
        row = idx // cols
        col = idx % cols
        x0 = offset_x + col * cell_w
        y0 = offset_y + row * cell_h
        bleed = min(int(round(cell_h * args.cell_bleed)), int(args.cell_bleed_max))
        ex0 = max(0, x0 - bleed)
        ey0 = max(0, y0 - bleed)
        ex1 = min(img.width, x0 + cell_w + bleed)
        ey1 = min(img.height, y0 + cell_h + bleed)
        cell_img = img.crop((ex0, ey0, ex1, ey1))
        contour_invert = invert
        if args.clean_components:
            cell_img = clean_cell_components(
                cell_img=cell_img,
                invert=invert,
                threshold=args.threshold,
                keep_components=component_limit_for_char(ch, args.keep_components),
                min_component_area=args.min_component_area,
                center_bias=args.component_center_bias,
                core_box=(x0 - ex0, y0 - ey0, x0 - ex0 + cell_w, y0 - ey0 + cell_h),
            )
            contour_invert = invert

        contours, _ = contours_from_image(
            cell_img,
            level=args.contour_level,
            simplify=args.simplify,
            trace_scale=args.trace_scale,
            trace_blur=args.trace_blur,
            smooth_iters=args.smooth_iters,
            invert=contour_invert,
        )
        if ex0 != x0 or ey0 != y0:
            dx = float(ex0 - x0)
            dy = float(ey0 - y0)
            shifted: list[np.ndarray] = []
            for contour in contours:
                pts = contour.copy()
                pts[:, 0] += dx
                pts[:, 1] += dy
                shifted.append(pts)
            contours = shifted

        baseline_px = float(row_baselines.get(row, cell_h * baseline_ratio_used))
        y_shift_px = float(cell_h * args.descender_lift) if ch in descender_chars else 0.0

        bounds = contours_bounds(contours)
        if bounds is not None:
            _, min_y, _, max_y = bounds
            min_y -= y_shift_px
            max_y -= y_shift_px
            max_above = max(max_above, max(0.0, baseline_px - min_y))
            max_below = max(max_below, max(0.0, max_y - baseline_px))

        glyph_entries.append(
            {
                "char": ch,
                "contours": contours,
                "baseline_px": baseline_px,
                "y_shift_px": y_shift_px,
            }
        )

    fallback_scale = args.upm / float(cell_h) * (1.0 - args.padding)
    padding_units = int(args.upm * args.padding)
    usable_upm = args.upm - 2 * padding_units
    total_height = max_above + max_below

    if total_height > 0 and usable_upm > 0:
        scale = usable_upm / total_height
        ascent = max(1, int(max_above * scale + padding_units))
        descent = -max(1, int(max_below * scale + padding_units))
    else:
        scale = fallback_scale
        ascent = max(1, int(args.upm * baseline_ratio_used))
        descent = -max(1, int(args.upm * (1.0 - baseline_ratio_used)))

    glyph_post_scale = args.glyph_post_scale if args.glyph_post_scale is not None else 1.0
    if glyph_post_scale != 1.0:
        scale *= glyph_post_scale
        ascent = max(1, int(ascent * glyph_post_scale))
        descent = -max(1, int(abs(descent) * glyph_post_scale))

    fb = FontBuilder(args.upm, isTTF=True)
    glyph_order = [".notdef"] + list(args.charset)
    has_space = " " in args.charset
    if not has_space:
        glyph_order.append("space")
    fb.setupGlyphOrder(glyph_order)
    fb.setupNameTable(
        dict(
            familyName=args.font_name,
            styleName="Regular",
            uniqueFontIdentifier=args.font_name,
            fullName=args.font_name,
            psName=args.font_name.replace(" ", "-"),
            version="Version 1.0",
        )
    )
    fb.setupPost()

    glyph_dict = {}
    metrics = {}

    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, args.upm))
    pen.lineTo((450, args.upm))
    pen.lineTo((450, 0))
    pen.closePath()
    glyph_dict[".notdef"] = pen.glyph()
    metrics[".notdef"] = (500, 50)

    for glyph_entry in glyph_entries:
        ch = glyph_entry["char"]
        contours = glyph_entry["contours"]

        pen = TTGlyphPen(None)
        advance = draw_glyph_fixed_grid(
            pen,
            contours,
            None,
            scale,
            args.side_bearing,
            cell_h,
            glyph_entry["baseline_px"],
            cell_w,
            y_shift_px=glyph_entry["y_shift_px"],
        )

        glyph_dict[ch] = pen.glyph()
        metrics[ch] = (advance, args.side_bearing)

    if not has_space:
        space_pen = TTGlyphPen(None)
        glyph_dict["space"] = space_pen.glyph()
        space_advance = max(int(args.upm * 0.33), args.side_bearing * 2)
        metrics["space"] = (space_advance, 0)

    fb.setupGlyf(glyph_dict)
    cmap = {ord(c): c for c in args.charset}
    if not has_space:
        cmap[ord(" ")] = "space"
    fb.setupCharacterMap(cmap)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupOS2(
        sTypoAscender=ascent,
        sTypoDescender=descent,
        usWinAscent=ascent,
        usWinDescent=abs(descent),
    )

    out_path = output_dir / f"{args.font_name.replace(' ', '_')}.ttf"
    fb.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
