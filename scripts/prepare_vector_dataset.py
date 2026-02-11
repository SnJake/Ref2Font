# --- START OF FILE prepare_vector_dataset.py ---
import argparse
import json
import math
import re
import sys
import numpy as np
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.filterPen import DecomposingPen
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
try:
    from fontnn.tokenizer import SVGTokenizer
except ImportError:
    sys.path.append(str(Path.cwd() / "src"))
    from fontnn.tokenizer import SVGTokenizer

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"

class DecomposingQu2CuPen(DecomposingPen):
    def __init__(self, glyph_set, out_pen, max_err, all_cubic=True):
        self.converter = Qu2CuPen(out_pen, max_err=max_err, all_cubic=all_cubic)
        super().__init__(glyph_set)

    def moveTo(self, p0):
        self.converter.moveTo(p0)

    def lineTo(self, p1):
        self.converter.lineTo(p1)

    def curveTo(self, *points):
        self.converter.curveTo(*points)

    def qCurveTo(self, *points):
        self.converter.qCurveTo(*points)

    def closePath(self):
        self.converter.closePath()

    def endPath(self):
        self.converter.endPath()

class DecomposingRecordingPen(DecomposingPen, RecordingPen):
    def __init__(self, glyph_set):
        RecordingPen.__init__(self)
        DecomposingPen.__init__(self, glyph_set)

def get_glyph_metrics(font, glyph_name):
    """Получаем метрики для правильной нормализации."""
    upm = font['head'].unitsPerEm
    
    # Вертикальные метрики
    os2 = font['OS/2']
    ascender = os2.sTypoAscender
    descender = os2.sTypoDescender
    
    # Высота "строки"
    total_height = ascender - descender
    
    # Если метрики битые (бывает в плохих шрифтах), фоллбэк на UPM
    if total_height == 0:
        total_height = upm
        ascender = upm
        descender = 0
        
    return {
        "upm": upm,
        "ascender": ascender,
        "descender": descender,
        "total_height": total_height
    }

def normalize_path(path_str, metrics):
    """
    Парсит SVG путь, нормализует координаты в 0..1
    относительно типографского окна (ascender/descender).
    """
    # Мы хотим вписать (descender...ascender) в диапазон (0.1...0.9) по Y,
    # чтобы оставить место для вылетов.
    # По X просто масштабируем пропорционально.
    
    scale = 0.8 / metrics['total_height']
    offset_y = -metrics['descender'] # Сдвигаем дно вверх
    
    # Центрирование по Y в окне 0..1:
    # Y_norm = (Y + offset_y) * scale + 0.1
    
    def transform_coord(val, is_y):
        if is_y:
            # SVGPathPen уже даёт экранные координаты (Y вниз)
            return (metrics['ascender'] - val) * scale + 0.1
        else:
            return val * scale + 0.1

    # Регулярка для поиска пар чисел
    # SVG команды: M x y, L x y, Q x1 y1 x2 y2 ...
    # Нам нужно найти все числа и понять, X это или Y.
    
    commands = re.split(r'([a-zA-Z])', path_str)
    new_commands = []
    
    for item in commands:
        if not item: continue
        if re.match(r'[a-zA-Z]', item):
            new_commands.append(item)
        else:
            # Это числа. Разбираем их
            coords = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', item)]
            norm_coords = []
            for i, val in enumerate(coords):
                # В SVG командах координаты обычно идут парами X, Y
                # (кроме H и V, но SVGPathPen обычно выдает полные координаты)
                is_y = (i % 2 == 1)
                n_val = transform_coord(val, is_y)
                # Округляем
                norm_coords.append(f"{n_val:.3f}")
            new_commands.append(" ".join(norm_coords))
            
    return "".join(new_commands)

def _recording_to_contours(ops):
    contours = []
    current_start = None
    current_segments = []
    current_pos = None

    def flush(closed):
        nonlocal current_start, current_segments, current_pos
        if current_start is None:
            return
        contours.append({
            "start": current_start,
            "segments": current_segments,
            "closed": closed,
        })
        current_start = None
        current_segments = []
        current_pos = None

    for op, pts in ops:
        if op == "moveTo":
            if current_start is not None:
                flush(False)
            current_start = pts[0]
            current_pos = current_start
        elif op == "lineTo":
            if current_start is None:
                continue
            end = pts[0]
            current_segments.append({"cmd": "L", "points": [end]})
            current_pos = end
        elif op == "curveTo":
            if current_start is None:
                continue
            c1, c2, end = pts
            current_segments.append({"cmd": "C", "points": [c1, c2, end]})
            current_pos = end
        elif op == "qCurveTo":
            if current_start is None or not pts:
                continue
            points = list(pts)
            if points[-1] is None:
                if len(points) < 2:
                    continue
                implied = (
                    (points[-2][0] + current_start[0]) / 2.0,
                    (points[-2][1] + current_start[1]) / 2.0,
                )
                points[-1] = implied
            end_on = points[-1]
            off = points[:-1]
            if not off:
                current_segments.append({"cmd": "L", "points": [end_on]})
                current_pos = end_on
                continue
            prev_on = current_pos if current_pos is not None else current_start
            if len(off) == 1:
                ctrl = off[0]
                c1 = (
                    prev_on[0] + (2.0 / 3.0) * (ctrl[0] - prev_on[0]),
                    prev_on[1] + (2.0 / 3.0) * (ctrl[1] - prev_on[1]),
                )
                c2 = (
                    end_on[0] + (2.0 / 3.0) * (ctrl[0] - end_on[0]),
                    end_on[1] + (2.0 / 3.0) * (ctrl[1] - end_on[1]),
                )
                current_segments.append({"cmd": "C", "points": [c1, c2, end_on]})
                current_pos = end_on
            else:
                prev = prev_on
                for i in range(len(off) - 1):
                    ctrl = off[i]
                    implied = (
                        (off[i][0] + off[i + 1][0]) / 2.0,
                        (off[i][1] + off[i + 1][1]) / 2.0,
                    )
                    c1 = (
                        prev[0] + (2.0 / 3.0) * (ctrl[0] - prev[0]),
                        prev[1] + (2.0 / 3.0) * (ctrl[1] - prev[1]),
                    )
                    c2 = (
                        implied[0] + (2.0 / 3.0) * (ctrl[0] - implied[0]),
                        implied[1] + (2.0 / 3.0) * (ctrl[1] - implied[1]),
                    )
                    current_segments.append({"cmd": "C", "points": [c1, c2, implied]})
                    prev = implied
                ctrl = off[-1]
                c1 = (
                    prev[0] + (2.0 / 3.0) * (ctrl[0] - prev[0]),
                    prev[1] + (2.0 / 3.0) * (ctrl[1] - prev[1]),
                )
                c2 = (
                    end_on[0] + (2.0 / 3.0) * (ctrl[0] - end_on[0]),
                    end_on[1] + (2.0 / 3.0) * (ctrl[1] - end_on[1]),
                )
                current_segments.append({"cmd": "C", "points": [c1, c2, end_on]})
                current_pos = end_on
        elif op == "closePath":
            flush(True)
        elif op == "endPath":
            flush(False)
        else:
            continue

    flush(False)
    return contours

def _get_cubic_contours(glyph_set, glyph_name, upm):
    recording_pen = RecordingPen()
    max_err = max(1.0, upm / 1000.0)
    pen = DecomposingQu2CuPen(glyph_set, recording_pen, max_err=max_err, all_cubic=True)
    try:
        glyph_set[glyph_name].draw(pen)
        ops = recording_pen.value
    except NotImplementedError:
        fallback_pen = DecomposingRecordingPen(glyph_set)
        glyph_set[glyph_name].draw(fallback_pen)
        ops = fallback_pen.value
    return _recording_to_contours(ops)

def _normalize_contours(contours, metrics):
    scale = 0.8 / metrics["total_height"]
    ascender = metrics["ascender"]

    def norm(p):
        return (p[0] * scale + 0.1, (ascender - p[1]) * scale + 0.1)

    out = []
    for contour in contours:
        segments = []
        for seg in contour["segments"]:
            segments.append({
                "cmd": seg["cmd"],
                "points": [norm(p) for p in seg["points"]],
            })
        out.append({
            "start": norm(contour["start"]),
            "segments": segments,
            "closed": contour["closed"],
        })
    return out

def _cubic_point(p0, p1, p2, p3, t):
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    a = mt2 * mt
    b = 3.0 * mt2 * t
    c = 3.0 * mt * t2
    d = t2 * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )

def _flatten_contour(contour, steps=8):
    points = []
    if not contour["segments"]:
        return points
    current = contour["start"]
    points.append(current)
    for seg in contour["segments"]:
        if seg["cmd"] == "L":
            end = seg["points"][0]
            points.append(end)
            current = end
        elif seg["cmd"] == "C":
            c1, c2, end = seg["points"]
            for i in range(1, steps + 1):
                t = i / steps
                points.append(_cubic_point(current, c1, c2, end, t))
            current = end
        else:
            end = seg["points"][-1]
            points.append(end)
            current = end
    return points

def _signed_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0

def _point_in_polygon(point, polygon):
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)):
            denom = (yj - yi) if (yj - yi) != 0 else 1e-12
            x_intersect = (xj - xi) * (y - yi) / denom + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside

def _reverse_contour(contour):
    if not contour["segments"]:
        return contour
    segments = []
    current = contour["start"]
    for seg in contour["segments"]:
        end = seg["points"][-1]
        segments.append({
            "cmd": seg["cmd"],
            "start": current,
            "points": seg["points"],
            "end": end,
        })
        current = end

    new_segments = []
    for seg in reversed(segments):
        if seg["cmd"] == "L":
            new_segments.append({"cmd": "L", "points": [seg["start"]]})
        elif seg["cmd"] == "C":
            c1, c2, _end = seg["points"]
            new_segments.append({"cmd": "C", "points": [c2, c1, seg["start"]]})
        else:
            new_segments.append({"cmd": "L", "points": [seg["start"]]})

    contour["start"] = segments[-1]["end"]
    contour["segments"] = new_segments
    return contour

def _rotate_contour_start(contour):
    if not contour["segments"]:
        return contour
    nodes = [contour["start"]] + [seg["points"][-1] for seg in contour["segments"]]
    if len(nodes) <= 1:
        return contour
    candidates = nodes[:-1]

    best_idx = 0
    best = candidates[0]
    for idx, pt in enumerate(candidates):
        if pt[0] < best[0] or (math.isclose(pt[0], best[0], abs_tol=1e-9) and pt[1] < best[1]):
            best = pt
            best_idx = idx

    if best_idx == 0:
        return contour

    contour["start"] = nodes[best_idx]
    contour["segments"] = contour["segments"][best_idx:] + contour["segments"][:best_idx]
    return contour

def _canonicalize_contours(contours):
    infos = []
    for contour in contours:
        if not contour["closed"] or not contour["segments"]:
            infos.append(None)
            continue
        poly = _flatten_contour(contour)
        if len(poly) < 3:
            infos.append(None)
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        infos.append({
            "poly": poly,
            "area": _signed_area(poly),
            "bbox": bbox,
            "centroid": centroid,
        })

    hole_flags = [False] * len(contours)
    for i, info in enumerate(infos):
        if info is None:
            continue
        inside = 0
        bbox = info["bbox"]
        bbox_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        test_points = [info["centroid"], bbox_center]
        for j, other in enumerate(infos):
            if i == j or other is None:
                continue
            if any(_point_in_polygon(tp, other["poly"]) for tp in test_points):
                inside += 1
        hole_flags[i] = (inside % 2 == 1)

    for i, contour in enumerate(contours):
        info = infos[i]
        if info is None:
            continue
        area = info["area"]
        is_hole = hole_flags[i]
        # In the normalized (y-down) space, positive area means clockwise.
        if is_hole:
            if area > 0:
                contours[i] = _reverse_contour(contour)
        else:
            if area < 0:
                contours[i] = _reverse_contour(contour)

    for i, contour in enumerate(contours):
        if contour["closed"]:
            contours[i] = _rotate_contour_start(contour)

    return contours

def _contours_to_svg(contours):
    if not contours:
        return ""

    def fmt(p):
        return f"{p[0]:.3f} {p[1]:.3f}"

    parts = []
    for contour in contours:
        parts.append(f"M {fmt(contour['start'])}")
        for seg in contour["segments"]:
            if seg["cmd"] == "L":
                parts.append(f"L {fmt(seg['points'][0])}")
            elif seg["cmd"] == "C":
                p1, p2, p3 = seg["points"]
                parts.append(f"C {fmt(p1)} {fmt(p2)} {fmt(p3)}")
        if contour["closed"]:
            parts.append("Z")
    return " ".join(parts)

def get_svg_data(font_path, char):
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        
        if ord(char) not in cmap: return None
        glyph_name = cmap[ord(char)]
        glyph_set = font.getGlyphSet()
        
        if glyph_name not in glyph_set: return None

        metrics = get_glyph_metrics(font, glyph_name)
        contours = _get_cubic_contours(glyph_set, glyph_name, metrics["upm"])
        if not contours:
            return None

        norm_contours = _normalize_contours(contours, metrics)
        canon_contours = _canonicalize_contours(norm_contours)
        path_str = _contours_to_svg(canon_contours)
        return path_str if path_str else None
        
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=1024)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.index.exists():
        print("Index not found")
        return

    with args.index.open("r", encoding="utf-8") as fp:
        fonts_meta = json.load(fp)
        
    tokenizer = SVGTokenizer(args.grid_size)
    data_samples = []
    
    print(f"Processing {len(fonts_meta)} fonts...")
    
    # Можно использовать ProcessPoolExecutor для ускорения
    for entry in tqdm(fonts_meta):
        font_path = Path(entry["path"])
        if not font_path.exists(): continue
            
        processed_glyphs = {}
        
        for char in CHARSET:
            norm_path = get_svg_data(font_path, char)
            if norm_path:
                tokens = tokenizer.encode(norm_path)
                # Фильтруем слишком длинные или пустые
                if 2 < len(tokens) < 2048: # Было 512
                    processed_glyphs[char] = tokens.numpy().tolist()
        
        if len(processed_glyphs) > 20: # Хотя бы 20 букв
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', f"{entry['style_key']}_{font_path.stem}")
            sample_file = args.output_dir / f"{safe_name}.json"
            
            with open(sample_file, 'w') as f:
                json.dump({
                    "font_path": str(font_path),
                    "glyphs": processed_glyphs
                }, f)
            data_samples.append(str(sample_file))

    with open(args.output_dir / "manifest.json", 'w') as f:
        json.dump(data_samples, f)

if __name__ == "__main__":
    main()
