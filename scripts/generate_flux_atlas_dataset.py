import argparse
import concurrent.futures as futures
import json
import math
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"
DEFAULT_INDEX = Path(r"G:\Programs\FontNN\data\raw\fonts_index.json")
DEFAULT_PROCESSED_DIR = Path(r"G:\Programs\FontNN\data\processed")


try:
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - optional safety
    TTFont = None


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = value.strip("_")
    return value or "font"


def unique_name(base: str, used: dict) -> str:
    if base not in used:
        used[base] = 1
        return base
    used[base] += 1
    return f"{base}_{used[base]}"


def normalize_glyphs(glyphs) -> set[str] | None:
    if isinstance(glyphs, dict):
        return {str(k) for k in glyphs.keys()}
    if isinstance(glyphs, list):
        return {str(k) for k in glyphs}
    return None


def load_font_entries(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    entries: list[dict] = []

    def add_entry(path_value, glyphs):
        if not path_value:
            return
        p = Path(path_value)
        if not p.is_absolute():
            p = (json_path.parent / p).resolve()
        if p.suffix.lower() not in {".ttf", ".otf"}:
            return
        entries.append({"path": p, "glyphs": normalize_glyphs(glyphs)})

    def handle_entry(entry):
        if isinstance(entry, str):
            add_entry(entry, None)
            return
        if isinstance(entry, dict):
            path_value = entry.get("path") or entry.get("font_path")
            glyphs = entry.get("glyphs")
            if path_value:
                add_entry(path_value, glyphs)
            if isinstance(entry.get("fonts"), list):
                for sub in entry["fonts"]:
                    handle_entry(sub)

    if isinstance(data, list):
        for item in data:
            handle_entry(item)
    elif isinstance(data, dict):
        if data.get("path") or data.get("font_path"):
            handle_entry(data)
        if isinstance(data.get("fonts"), list):
            for item in data["fonts"]:
                handle_entry(item)

    return entries


def gather_custom_jsons(processed_dir: Path) -> list[Path]:
    if not processed_dir.exists():
        return []
    custom_files = []
    for path in processed_dir.rglob("*.json"):
        if "custom" in path.name.lower():
            custom_files.append(path)
    return sorted(custom_files)


def dedupe_entries(entries: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for entry in entries:
        p: Path = entry["path"]
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        glyphs = entry.get("glyphs")
        if key not in merged:
            merged[key] = {"path": p, "glyphs": set(glyphs) if glyphs else None}
        else:
            existing = merged[key].get("glyphs")
            if glyphs:
                if existing:
                    existing.update(glyphs)
                else:
                    merged[key]["glyphs"] = set(glyphs)
    return list(merged.values())


def supports_charset(font_path: Path, charset: str) -> bool:
    if TTFont is None:
        return True
    try:
        font = TTFont(font_path)
    except Exception:
        return False
    try:
        cmap = font.getBestCmap() or {}
        for ch in charset:
            if ord(ch) not in cmap:
                return False
        return True
    finally:
        try:
            font.close()
        except Exception:
            pass


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


def max_bbox_for_charset(font: ImageFont.FreeTypeFont, charset: str) -> tuple[int, int] | None:
    max_w = 0
    max_h = 0
    for ch in charset:
        bbox = font.getbbox(ch)
        if not bbox:
            return None
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return None
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h
    return max_w, max_h


def glyph_metrics(font: ImageFont.FreeTypeFont, ch: str) -> tuple[int, int, tuple[int, int, int, int]] | None:
    bbox = font.getbbox(ch)
    if not bbox:
        return None
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    return w, h, bbox


def pair_metrics(
    font: ImageFont.FreeTypeFont, left: str, right: str, gap_ratio: float
) -> tuple[int, int, int, tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    left_metrics = glyph_metrics(font, left)
    right_metrics = glyph_metrics(font, right)
    if left_metrics is None or right_metrics is None:
        return None
    left_w, left_h, left_bbox = left_metrics
    right_w, right_h, right_bbox = right_metrics
    gap = max(4, int(max(left_w, right_w) * gap_ratio))
    total_w = left_w + gap + right_w
    max_h = max(left_h, right_h)
    return total_w, max_h, gap, left_bbox, right_bbox


def choose_font(font_path: Path, charset: str, cell_size: int, fill_ratio: float = 0.85) -> ImageFont.FreeTypeFont | None:
    min_size = 4
    max_size = max(min_size, int(cell_size * 0.95))
    target = int(cell_size * fill_ratio)
    best_size = None

    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(str(font_path), size=mid)
        except Exception:
            return None
        metrics = max_bbox_for_charset(font, charset)
        if metrics is None:
            return None
        max_w, max_h = metrics
        if max_w <= target and max_h <= target:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_size is None:
        return None
    try:
        return ImageFont.truetype(str(font_path), size=best_size)
    except Exception:
        return None


def choose_font_for_pair(
    font_path: Path,
    left: str,
    right: str,
    canvas_size: int,
    fill_ratio: float = 0.8,
    gap_ratio: float = 0.2,
) -> ImageFont.FreeTypeFont | None:
    min_size = 8
    max_size = max(min_size, int(canvas_size * 0.9))
    target = int(canvas_size * fill_ratio)
    best_size = None

    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(str(font_path), size=mid)
        except Exception:
            return None
        metrics = pair_metrics(font, left, right, gap_ratio)
        if metrics is None:
            return None
        total_w, max_h, _, _, _ = metrics
        if total_w <= target and max_h <= target:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_size is None:
        return None
    try:
        return ImageFont.truetype(str(font_path), size=best_size)
    except Exception:
        return None


def draw_char(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, ch: str, cell_x: int, cell_y: int, cell_size: int) -> bool:
    bbox = font.getbbox(ch)
    if not bbox:
        return False
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return False
    x = cell_x + (cell_size - w) / 2 - bbox[0]
    y = cell_y + (cell_size - h) / 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=(255, 255, 255))
    return True


def draw_centered_pair(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    canvas_size: int,
    left: str,
    right: str,
    gap_ratio: float = 0.2,
) -> bool:
    metrics = pair_metrics(font, left, right, gap_ratio)
    if metrics is None:
        return False
    total_w, max_h, gap, left_bbox, right_bbox = metrics
    block_left = (canvas_size - total_w) / 2
    block_top = (canvas_size - max_h) / 2

    left_w = left_bbox[2] - left_bbox[0]
    left_h = left_bbox[3] - left_bbox[1]
    right_w = right_bbox[2] - right_bbox[0]
    right_h = right_bbox[3] - right_bbox[1]

    left_x = block_left - left_bbox[0]
    left_y = block_top + (max_h - left_h) / 2 - left_bbox[1]
    right_x = block_left + left_w + gap - right_bbox[0]
    right_y = block_top + (max_h - right_h) / 2 - right_bbox[1]

    draw.text((left_x, left_y), left, font=font, fill=(255, 255, 255))
    draw.text((right_x, right_y), right, font=font, fill=(255, 255, 255))
    return True


def render_target(
    font: ImageFont.FreeTypeFont,
    charset: str,
    canvas_size: int,
    grid: tuple[int, int, int, int, int],
) -> Image.Image | None:
    cols, rows, cell_size, offset_x, offset_y = grid
    target = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw_target = ImageDraw.Draw(target)

    for idx, ch in enumerate(charset):
        row = idx // cols
        col = idx % cols
        cell_x = offset_x + col * cell_size
        cell_y = offset_y + row * cell_size
        if not draw_char(draw_target, font, ch, cell_x, cell_y, cell_size):
            return None

    return target


def render_control(
    font: ImageFont.FreeTypeFont, canvas_size: int, left: str = "A", right: str = "a"
) -> Image.Image | None:
    control = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw_control = ImageDraw.Draw(control)
    if not draw_centered_pair(draw_control, font, canvas_size, left, right):
        return None
    return control


def save_webp(image: Image.Image, path: Path) -> None:
    image.save(path, format="WEBP", lossless=True, quality=100, method=6)


def process_font(task: dict) -> tuple[str, str | None]:
    font_path: Path = task["font_path"]
    safe_name: str = task["safe_name"]
    charset: str = task["charset"]
    canvas: int = task["canvas"]
    grid: tuple[int, int, int, int, int] = task["grid"]
    targets_dir: Path = task["targets_dir"]
    controls_dir: Path = task["controls_dir"]
    prompt_text: str = task["prompt_text"]
    glyphs: set[str] | None = task.get("glyphs")

    target_path = targets_dir / f"{safe_name}.webp"
    control_path = controls_dir / f"{safe_name}.webp"
    prompt_path = targets_dir / f"{safe_name}.txt"

    try:
        if glyphs is not None:
            if not set(charset).issubset(glyphs):
                return "skip", f"Skip (missing glyphs from json): {font_path}"
        else:
            if not supports_charset(font_path, charset):
                return "skip", f"Skip (missing glyphs): {font_path}"
        atlas_font = choose_font(font_path, charset, grid[2])
        if atlas_font is None:
            return "skip", f"Skip (render error): {font_path}"
        target = render_target(atlas_font, charset, canvas, grid)
        if target is None:
            return "skip", f"Skip (bbox error): {font_path}"
        control_font = choose_font_for_pair(font_path, "A", "a", canvas)
        if control_font is None:
            return "skip", f"Skip (control render error): {font_path}"
        control = render_control(control_font, canvas)
        if control is None:
            return "skip", f"Skip (control bbox error): {font_path}"

        save_webp(target, target_path)
        save_webp(control, control_path)
        prompt_path.write_text(prompt_text, encoding="utf-8")
        return "ok", None
    except Exception as exc:
        return "skip", f"Skip (exception): {font_path} ({exc})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate atlas pairs (target/control) for FLUX.2 Klein 9B training."
    )
    parser.add_argument(
        "--fonts-index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Path to fonts_index.json (list of font paths).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for targets/controls.",
    )
    parser.add_argument(
        "--canvas",
        type=int,
        default=2048,
        help="Canvas size (square).",
    )
    parser.add_argument(
        "--charset",
        type=str,
        default=CHARSET,
        help="Characters to render into the atlas.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory with processed jsons (custom).",
    )
    parser.add_argument(
        "--skip-custom",
        action="store_true",
        help="Skip loading custom jsons from processed dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel workers (set 1 to disable multiprocessing).",
    )
    args = parser.parse_args()

    if not args.charset:
        raise SystemExit("charset is empty")

    if args.output_dir is None:
        raw = input("Output directory: ").strip()
        if not raw:
            raise SystemExit("output_dir is required")
        output_dir = Path(raw)
    else:
        output_dir = args.output_dir

    fonts_index = args.fonts_index
    if not fonts_index.exists():
        raise SystemExit(f"fonts_index.json not found: {fonts_index}")

    entries = load_font_entries(fonts_index)
    custom_jsons = []
    if not args.skip_custom and args.processed_dir is not None:
        custom_jsons = gather_custom_jsons(args.processed_dir)
        for json_path in custom_jsons:
            entries.extend(load_font_entries(json_path))
    entries = dedupe_entries(entries)
    if not entries:
        raise SystemExit("No font entries found in JSONs")

    targets_dir = output_dir / "targets"
    controls_dir = output_dir / "controls"
    targets_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)

    grid = compute_grid(len(args.charset), args.canvas, args.canvas)
    prompt_text = (
        'Generate letters and symbols "' + args.charset + '" in the style of the letters given to you as a reference.'
    )
    used_names: dict[str, int] = {}
    processed = 0
    skipped = 0

    if custom_jsons:
        print(f"Loaded {len(custom_jsons)} custom jsons from {args.processed_dir}")

    tasks: list[dict] = []
    for entry in entries:
        font_path = entry["path"]
        glyphs = entry.get("glyphs")
        if not font_path.exists():
            skipped += 1
            continue
        safe_base = sanitize_name(font_path.stem)
        safe_name = unique_name(safe_base, used_names)
        tasks.append(
            {
                "font_path": font_path,
                "safe_name": safe_name,
                "charset": args.charset,
                "canvas": args.canvas,
                "grid": grid,
                "targets_dir": targets_dir,
                "controls_dir": controls_dir,
                "prompt_text": prompt_text,
                "glyphs": glyphs,
            }
        )

    if args.workers < 1:
        args.workers = 1

    if args.workers <= 1:
        for task in tqdm(tasks, desc="Rendering fonts"):
            status, message = process_font(task)
            if status == "ok":
                processed += 1
            else:
                skipped += 1
                if message:
                    tqdm.write(message)
    else:
        with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_list = [executor.submit(process_font, task) for task in tasks]
            with tqdm(total=len(future_list), desc="Rendering fonts") as pbar:
                for future in futures.as_completed(future_list):
                    status, message = future.result()
                    if status == "ok":
                        processed += 1
                    else:
                        skipped += 1
                        if message:
                            tqdm.write(message)
                    pbar.update(1)

    print(f"Done. Processed: {processed}. Skipped: {skipped}.")


if __name__ == "__main__":
    main()
