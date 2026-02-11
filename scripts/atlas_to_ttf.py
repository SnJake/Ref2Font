import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


LATIN_CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"&'
CYRILLIC_CHARSET = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    '0123456789!?.,;:-"&'
)
PRESET_CHARSETS: dict[str, str] = {
    "latin": LATIN_CHARSET,
    "cyrillic": CYRILLIC_CHARSET,
}


@dataclass
class Segment:
    row_idx: int
    x0: int
    x1: int
    y0: int
    y1: int


@dataclass
class GlyphInfo:
    char: str
    row_idx: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    baseline: float


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def segment_row_simple(proj: np.ndarray, segments: int, min_width: int) -> list[int]:
    width = len(proj)
    if segments <= 1:
        return []
    if width < segments * min_width:
        step = width / segments
        cuts = [int(round(step * i)) - 1 for i in range(1, segments)]
        cuts = [min(max(c, 0), width - 2) for c in cuts]
        return sorted(set(cuts))

    cut_cost = proj[:-1].astype(float)
    cuts_needed = segments - 1
    dp_prev = np.full(width - 1, np.inf)
    prevs = [None] * (cuts_needed + 1)

    min_i = min_width - 1
    max_i = width - min_width * (segments - 1) - 1
    for i in range(min_i, max_i + 1):
        dp_prev[i] = cut_cost[i]
    prevs[1] = np.full(width - 1, -1, dtype=int)

    for c in range(2, cuts_needed + 1):
        dp_curr = np.full(width - 1, np.inf)
        prev = np.full(width - 1, -1, dtype=int)
        best_val = np.inf
        best_idx = -1
        min_i = c * min_width - 1
        max_i = width - min_width * (segments - c) - 1
        for i in range(min_i, max_i + 1):
            j = i - min_width
            if j >= 0 and dp_prev[j] < best_val:
                best_val = dp_prev[j]
                best_idx = j
            if best_idx != -1:
                dp_curr[i] = cut_cost[i] + best_val
                prev[i] = best_idx
        dp_prev = dp_curr
        prevs[c] = prev

    valid = np.isfinite(dp_prev)
    if not valid.any():
        return []
    last_i = int(np.argmin(np.where(valid, dp_prev, np.inf)))
    cuts = [last_i]
    for c in range(cuts_needed, 1, -1):
        last_i = prevs[c][last_i]
        cuts.append(last_i)
    cuts.reverse()
    return cuts


def adjust_counts(counts: list[int], widths: list[int], target: int, median_width: float) -> list[int]:
    total = sum(counts)
    ideal = [w / median_width for w in widths]
    while total < target:
        deltas = [ideal[i] - counts[i] for i in range(len(counts))]
        idx = int(np.argmax(deltas))
        counts[idx] += 1
        total += 1
    while total > target:
        deltas = [counts[i] - ideal[i] for i in range(len(counts))]
        best = -1.0
        idx = None
        for i, delta in enumerate(deltas):
            if counts[i] > 1 and delta > best:
                best = delta
                idx = i
        if idx is None:
            break
        counts[idx] -= 1
        total -= 1
    return counts


def mask_to_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    active: dict[tuple[int, int], tuple[int, int]] = {}
    rects: list[tuple[int, int, int, int]] = []
    for y in range(h):
        row = mask[y]
        runs = []
        x = 0
        while x < w:
            if row[x]:
                start = x
                while x < w and row[x]:
                    x += 1
                runs.append((start, x))
            else:
                x += 1
        new_active: dict[tuple[int, int], tuple[int, int]] = {}
        run_set = set(runs)
        for run in runs:
            if run in active:
                start_y, _ = active[run]
                new_active[run] = (start_y, y)
            else:
                new_active[run] = (y, y)
        for run, (start_y, last_y) in active.items():
            if run not in run_set:
                x0, x1 = run
                rects.append((x0, start_y, x1, last_y + 1))
        active = new_active
    for run, (start_y, last_y) in active.items():
        x0, x1 = run
        rects.append((x0, start_y, x1, last_y + 1))
    return rects


def mask_visual_centroid_x(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()) + 0.5


def draw_debug(image: Image.Image, glyphs: list[GlyphInfo], out_path: Path) -> None:
    if not glyphs:
        return
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 140, 0)]
    for idx, glyph in enumerate(glyphs):
        x0, y0, x1, y1 = glyph.bbox
        color = colors[idx % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), glyph.char, fill=color)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice a glyph atlas and build a Windows-installable TTF font."
    )
    parser.add_argument("--image", required=True, help="Path to atlas image (white on black or black on white).")
    parser.add_argument("--output-dir", required=True, help="Output directory for the generated TTF.")
    parser.add_argument("--font-name", default="AtlasFont", help="Font family name.")
    parser.add_argument(
        "--language",
        choices=["latin", "cyrillic"],
        default="latin",
        help="Preset charset language used when --charset is not provided.",
    )
    parser.add_argument("--charset", default=None, help="Characters in reading order (overrides --language).")
    parser.add_argument("--threshold", type=int, default=127, help="Binarization threshold (0-255).")
    parser.add_argument("--row-gap", type=int, default=2, help="Max empty row gap to merge row bands.")
    parser.add_argument("--gap-threshold", type=int, default=0, help="Column projection <= value treated as gap.")
    parser.add_argument("--min-width-ratio", type=float, default=0.2, help="Min split width as fraction of median.")
    parser.add_argument("--downsample", type=float, default=4.0, help="Downsample factor for outlines (>=1).")
    parser.add_argument("--upm", type=int, default=1024, help="Units per em.")
    parser.add_argument("--padding", type=float, default=0.05, help="Vertical padding ratio in em units.")
    parser.add_argument("--side-bearing", type=int, default=50, help="Left/right side bearing in font units.")
    parser.add_argument(
        "--align-mode",
        choices=["geometric", "visual"],
        default="geometric",
        help="Horizontal alignment mode: geometric bbox-left or visual centroid-center.",
    )
    parser.add_argument("--debug-dir", default=None, help="Optional directory to save debug images.")
    args = parser.parse_args()
    if args.charset is None:
        args.charset = PRESET_CHARSETS[args.language]

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    mean_val = float(arr.mean())
    if mean_val < 128:
        mask = arr > args.threshold
    else:
        mask = arr < args.threshold

    rows = mask.any(axis=1)
    row_runs = merge_runs(find_runs(rows), max_gap=args.row_gap)
    if not row_runs:
        raise SystemExit("No glyph rows detected. Check threshold/background.")

    segments: list[Segment] = []
    for row_idx, (y0, y1) in enumerate(row_runs):
        submask = mask[y0 : y1 + 1]
        proj = submask.sum(axis=0)
        xs = np.where(proj > 0)[0]
        if xs.size == 0:
            continue
        x_min, x_max = int(xs[0]), int(xs[-1])
        gaps = proj <= args.gap_threshold
        gap_runs = find_runs(gaps)
        start = x_min
        for s, e in gap_runs:
            if s <= x_min or s >= x_max:
                continue
            if s > start:
                segments.append(Segment(row_idx, start, s - 1, y0, y1))
            start = e + 1
        if start <= x_max:
            segments.append(Segment(row_idx, start, x_max, y0, y1))

    if not segments:
        raise SystemExit("No glyph segments found. Try lowering --gap-threshold.")

    widths = [seg.x1 - seg.x0 + 1 for seg in segments]
    median_width = float(np.median(widths)) if widths else 1.0
    counts = [max(1, int(round(w / median_width))) for w in widths]
    counts = adjust_counts(counts, widths, len(args.charset), median_width)
    if sum(counts) != len(args.charset):
        raise SystemExit("Unable to match charset length with current segmentation.")

    min_width = max(5, int(median_width * args.min_width_ratio))
    glyph_boxes: list[Segment] = []
    for seg, n in zip(segments, counts):
        proj = mask[seg.y0 : seg.y1 + 1, seg.x0 : seg.x1 + 1].sum(axis=0)
        cuts = segment_row_simple(proj, n, min_width=min_width)
        start = 0
        width = seg.x1 - seg.x0 + 1
        for cut in cuts:
            glyph_boxes.append(Segment(seg.row_idx, seg.x0 + start, seg.x0 + cut, seg.y0, seg.y1))
            start = cut + 1
        glyph_boxes.append(Segment(seg.row_idx, seg.x0 + start, seg.x0 + width - 1, seg.y0, seg.y1))

    if len(glyph_boxes) != len(args.charset):
        raise SystemExit("Final glyph count does not match charset length.")

    # Baseline per row (median of glyph bottoms)
    row_baselines: dict[int, int] = {}
    for row_idx in range(len(row_runs)):
        bottoms = []
        for seg in glyph_boxes:
            if seg.row_idx != row_idx:
                continue
            region = mask[seg.y0 : seg.y1 + 1, seg.x0 : seg.x1 + 1]
            ys = np.where(region)[0]
            if ys.size == 0:
                continue
            bottoms.append(seg.y0 + int(ys.max()))
        if not bottoms:
            continue
        row_baselines[row_idx] = int(np.median(bottoms))

    glyphs: list[GlyphInfo] = []
    max_above = 0.0
    max_below = 0.0
    for char, seg in zip(args.charset, glyph_boxes):
        region = mask[seg.y0 : seg.y1 + 1, seg.x0 : seg.x1 + 1]
        if region.size == 0:
            continue
        orig_h, orig_w = region.shape
        scale_y = 1.0
        if args.downsample and args.downsample > 1:
            new_w = max(1, int(round(orig_w / args.downsample)))
            new_h = max(1, int(round(orig_h / args.downsample)))
            scale_y = new_h / float(orig_h)
            pil = Image.fromarray((region.astype(np.uint8) * 255))
            pil = pil.resize((new_w, new_h), Image.NEAREST)
            region = np.array(pil) > 127

        baseline = float(row_baselines.get(seg.row_idx, seg.y1))
        baseline = (baseline - seg.y0) * scale_y

        ys = np.where(region)[0]
        if ys.size == 0:
            glyphs.append(GlyphInfo(char, seg.row_idx, (seg.x0, seg.y0, seg.x1, seg.y1), region, baseline))
            continue
        top = float(ys.min())
        bottom = float(ys.max())
        max_above = max(max_above, baseline - top)
        max_below = max(max_below, bottom - baseline)

        glyphs.append(GlyphInfo(char, seg.row_idx, (seg.x0, seg.y0, seg.x1, seg.y1), region, baseline))

    if not glyphs:
        raise SystemExit("No glyphs extracted after processing.")

    padding_units = int(args.upm * args.padding)
    total_height = max_above + max_below
    scale = 1.0
    if total_height > 0:
        scale = (args.upm - 2 * padding_units) / total_height

    ascent = int(max_above * scale + padding_units)
    descent = -int(max_below * scale + padding_units)

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
    # Use post format 3.0 to avoid latin-1 glyph-name constraints.
    fb.setupPost(keepGlyphNames=False)

    glyph_dict = {}
    metrics = {}

    # .notdef glyph
    pen = TTGlyphPen(None)
    pen.moveTo((100, 100))
    pen.lineTo((100, 900))
    pen.lineTo((900, 900))
    pen.lineTo((900, 100))
    pen.closePath()
    glyph_dict[".notdef"] = pen.glyph()
    metrics[".notdef"] = (args.upm, 0)

    for glyph in glyphs:
        pen = TTGlyphPen(None)
        rects = mask_to_rects(glyph.mask)
        content_width = float(glyph.mask.shape[1])
        x_anchor = 0.0
        if args.align_mode == "visual":
            visual_center_x = mask_visual_centroid_x(glyph.mask)
            if visual_center_x is not None:
                x_anchor = float(visual_center_x) - (content_width * 0.5)
        for x0, y0, x1, y1 in rects:
            fx0 = (x0 - x_anchor) * scale + args.side_bearing
            fx1 = (x1 - x_anchor) * scale + args.side_bearing
            fy0 = (glyph.baseline - y0) * scale
            fy1 = (glyph.baseline - y1) * scale
            if fx0 == fx1 or fy0 == fy1:
                continue
            pen.moveTo((fx0, fy0))
            pen.lineTo((fx1, fy0))
            pen.lineTo((fx1, fy1))
            pen.lineTo((fx0, fy1))
            pen.closePath()
        glyph_dict[glyph.char] = pen.glyph()
        advance = int(glyph.mask.shape[1] * scale + 2 * args.side_bearing)
        metrics[glyph.char] = (advance, args.side_bearing)

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
    fb.setupOS2(sTypoAscender=ascent, sTypoDescender=descent, usWinAscent=ascent, usWinDescent=abs(descent))

    out_path = output_dir / f"{args.font_name.replace(' ', '_')}.ttf"
    fb.save(out_path)
    print(f"Saved TTF to {out_path}")

    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        draw_debug(img, glyphs, debug_dir / "segmentation.png")


if __name__ == "__main__":
    main()
