from __future__ import annotations

import argparse
import concurrent.futures as futures
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from skimage import filters, measure, morphology
except Exception as exc:
    raise SystemExit("scikit-image is required: pip install scikit-image") from exc


VALID_SUFFIXES = {".webp", ".png", ".jpg", ".jpeg"}
DEFAULT_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"


def compute_grid(count: int, width: int, height: int) -> tuple[int, int, int, int, int]:
    best = None
    best_grid = (1, count, min(width, height) // max(1, count), 0, 0)
    for cols in range(1, max(2, count + 1)):
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


def detect_background_is_white(gray: np.ndarray, border: int = 4) -> bool:
    h, w = gray.shape
    b = max(1, int(border))
    b = min(b, h, w)
    top = gray[:b, :].ravel()
    bottom = gray[-b:, :].ravel()
    left = gray[:, :b].ravel()
    right = gray[:, -b:].ravel()
    vals = np.concatenate([top, bottom, left, right]) if (top.size + bottom.size + left.size + right.size) else gray.ravel()
    return float(vals.mean()) > 127.5


def parse_charset_from_prompt(prompt_path: Path) -> str | None:
    if not prompt_path.exists():
        return None
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except Exception:
        return None
    # Expected prompt: Generate letters and symbols "<charset>" ...
    m = re.search(r'"([^"]+)"', text)
    if not m:
        return None
    value = m.group(1)
    return value if value else None


def build_binary_mask(gray: np.ndarray, threshold_mode: str, threshold: int) -> np.ndarray:
    bg_white = detect_background_is_white(gray)
    if threshold_mode == "otsu":
        try:
            thr = float(filters.threshold_otsu(gray))
        except Exception:
            thr = float(threshold)
    else:
        thr = float(threshold)

    if bg_white:
        return gray < thr
    return gray > thr


def clean_cell_components(
    cell_mask: np.ndarray,
    *,
    center_x_ratio: float,
    keep_components: int,
    min_component_area: int,
) -> np.ndarray:
    if not np.any(cell_mask):
        return cell_mask

    labels = measure.label(cell_mask, connectivity=2)
    if labels.max() <= 0:
        return np.zeros_like(cell_mask, dtype=bool)

    h, w = cell_mask.shape
    x0 = w * float(center_x_ratio)
    x1 = w * (1.0 - float(center_x_ratio))

    center_candidates: list[tuple[float, int]] = []
    fallback_candidates: list[tuple[float, int]] = []

    for region in measure.regionprops(labels):
        area = int(region.area)
        if area < int(min_component_area):
            continue
        minr, minc, maxr, maxc = region.bbox
        cy, cx = region.centroid
        overlaps_center = (maxc > x0) and (minc < x1)
        centerish = x0 <= cx <= x1
        dist_center = abs(cx - (w * 0.5))
        score = float(area) - 0.15 * float(dist_center)
        item = (score, int(region.label))
        if centerish or overlaps_center:
            center_candidates.append(item)
        else:
            fallback_candidates.append(item)

    selected = center_candidates if center_candidates else fallback_candidates
    if not selected:
        return np.zeros_like(cell_mask, dtype=bool)

    selected.sort(key=lambda x: x[0], reverse=True)
    keep_labels = {label for _, label in selected[: max(1, int(keep_components))]}
    cleaned = np.isin(labels, list(keep_labels))

    if not np.any(cleaned):
        # Fallback: keep the largest component.
        best = max(measure.regionprops(labels), key=lambda r: int(r.area))
        cleaned = labels == int(best.label)

    return cleaned


def clean_atlas_mask(
    binary: np.ndarray,
    *,
    cols: int,
    rows: int,
    cell: int,
    offset_x: int,
    offset_y: int,
    center_x_ratio: float,
    keep_components: int,
    min_component_area: int,
) -> np.ndarray:
    h, w = binary.shape
    out = np.zeros_like(binary, dtype=bool)
    for row in range(rows):
        for col in range(cols):
            x0 = offset_x + col * cell
            y0 = offset_y + row * cell
            x1 = min(w, x0 + cell)
            y1 = min(h, y0 + cell)
            if x0 >= x1 or y0 >= y1:
                continue

            cell_mask = binary[y0:y1, x0:x1]
            cleaned = clean_cell_components(
                cell_mask,
                center_x_ratio=center_x_ratio,
                keep_components=keep_components,
                min_component_area=min_component_area,
            )
            out[y0:y1, x0:x1] = cleaned
    return out


@dataclass
class Task:
    image_path: Path
    rel_path: Path
    output_path: Path
    prompt_path: Path
    fixed_charset: str | None
    fixed_charset_len: int | None
    threshold_mode: str
    threshold: int
    opening_radius: int
    force_cols: int | None
    force_rows: int | None
    center_x_ratio: float
    keep_components: int
    min_component_area: int


def process_task(task: Task) -> tuple[bool, str]:
    try:
        gray = np.asarray(Image.open(task.image_path).convert("L"), dtype=np.uint8)
    except Exception as exc:
        return False, f"Failed to open {task.image_path}: {exc}"

    binary = build_binary_mask(gray, threshold_mode=task.threshold_mode, threshold=task.threshold)
    if task.opening_radius > 0:
        selem = morphology.disk(int(task.opening_radius))
        binary = morphology.binary_opening(binary, footprint=selem)

    if task.force_cols and task.force_rows:
        cols = int(task.force_cols)
        rows = int(task.force_rows)
        cell = min(gray.shape[1] // cols, gray.shape[0] // rows)
        offset_x = (gray.shape[1] - cell * cols) // 2
        offset_y = (gray.shape[0] - cell * rows) // 2
    else:
        if task.fixed_charset:
            char_count = len(task.fixed_charset)
        elif task.fixed_charset_len and task.fixed_charset_len > 0:
            char_count = int(task.fixed_charset_len)
        else:
            parsed = parse_charset_from_prompt(task.prompt_path)
            char_count = len(parsed) if parsed else len(DEFAULT_CHARSET)
        cols, rows, cell, offset_x, offset_y = compute_grid(char_count, gray.shape[1], gray.shape[0])

    cleaned = clean_atlas_mask(
        binary,
        cols=cols,
        rows=rows,
        cell=cell,
        offset_x=offset_x,
        offset_y=offset_y,
        center_x_ratio=task.center_x_ratio,
        keep_components=task.keep_components,
        min_component_area=task.min_component_area,
    )
    out_img = Image.fromarray((cleaned.astype(np.uint8) * 255), mode="L")

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = task.output_path.suffix.lower()
    if suffix == ".webp":
        out_img.save(task.output_path, format="WEBP", lossless=True, quality=100, method=6)
    else:
        out_img.save(task.output_path, format="PNG")
    return True, str(task.rel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned atlas masks in dataset_root/masks from dataset_root/targets.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root containing targets/ and controls/.")
    parser.add_argument("--targets-subdir", type=str, default="targets", help="Targets subdirectory name.")
    parser.add_argument("--masks-subdir", type=str, default="masks", help="Output masks subdirectory name.")
    parser.add_argument("--suffix", choices=["source", "png", "webp"], default="source", help="Output image suffix.")
    parser.add_argument("--charset", type=str, default=None, help="Fixed charset (overrides prompt parsing).")
    parser.add_argument("--charset-len", type=int, default=None, help="Fixed charset length if prompt is unavailable.")
    parser.add_argument("--cols", type=int, default=None, help="Force grid columns.")
    parser.add_argument("--rows", type=int, default=None, help="Force grid rows.")
    parser.add_argument("--threshold-mode", choices=["fixed", "otsu"], default="otsu")
    parser.add_argument("--threshold", type=int, default=127, help="Threshold for fixed mode.")
    parser.add_argument("--opening-radius", type=int, default=0, help="Optional morphology opening radius in px.")
    parser.add_argument("--center-x-ratio", type=float, default=0.22, help="Horizontal center stripe half-margin ratio.")
    parser.add_argument("--keep-components", type=int, default=4, help="How many components per cell to keep.")
    parser.add_argument("--min-component-area", type=int, default=4, help="Minimum connected-component area in px.")
    parser.add_argument("--workers", type=int, default=0, help="Process pool workers. 0 or 1 = single-process.")
    args = parser.parse_args()

    if args.center_x_ratio < 0 or args.center_x_ratio >= 0.5:
        raise SystemExit("center-x-ratio must be in [0, 0.5)")
    if args.keep_components < 1:
        raise SystemExit("keep-components must be >= 1")
    if args.min_component_area < 1:
        raise SystemExit("min-component-area must be >= 1")
    if bool(args.cols) != bool(args.rows):
        raise SystemExit("Set both --cols and --rows together, or neither.")

    dataset_dir = args.dataset_dir
    targets_dir = dataset_dir / args.targets_subdir
    masks_dir = dataset_dir / args.masks_subdir
    if not targets_dir.exists():
        raise SystemExit(f"targets dir not found: {targets_dir}")

    image_paths = sorted([p for p in targets_dir.rglob("*") if p.suffix.lower() in VALID_SUFFIXES])
    if not image_paths:
        raise SystemExit(f"No target images found in: {targets_dir}")

    tasks: list[Task] = []
    for image_path in image_paths:
        rel = image_path.relative_to(targets_dir)
        if args.suffix == "source":
            out_rel = rel
        elif args.suffix == "webp":
            out_rel = rel.with_suffix(".webp")
        else:
            out_rel = rel.with_suffix(".png")
        tasks.append(
            Task(
                image_path=image_path,
                rel_path=rel,
                output_path=masks_dir / out_rel,
                prompt_path=image_path.with_suffix(".txt"),
                fixed_charset=args.charset,
                fixed_charset_len=args.charset_len,
                threshold_mode=args.threshold_mode,
                threshold=args.threshold,
                opening_radius=args.opening_radius,
                force_cols=args.cols,
                force_rows=args.rows,
                center_x_ratio=args.center_x_ratio,
                keep_components=args.keep_components,
                min_component_area=args.min_component_area,
            )
        )

    ok = 0
    failed = 0
    workers = int(args.workers)
    if workers <= 1:
        for task in tqdm(tasks, desc="Generating masks"):
            success, message = process_task(task)
            if success:
                ok += 1
            else:
                failed += 1
                tqdm.write(message)
    else:
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_list = [executor.submit(process_task, task) for task in tasks]
            with tqdm(total=len(future_list), desc="Generating masks") as pbar:
                for fut in futures.as_completed(future_list):
                    success, message = fut.result()
                    if success:
                        ok += 1
                    else:
                        failed += 1
                        tqdm.write(message)
                    pbar.update(1)

    print(f"Done. Masks: {ok}. Failed: {failed}. Output: {masks_dir}")


if __name__ == "__main__":
    main()

