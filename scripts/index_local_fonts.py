import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm.auto import tqdm


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        # METADATA.pb uses textproto escaping, best effort decode
        content = raw[1:-1]
        return bytes(content, "utf-8").decode("unicode_escape")
    if raw.isdigit():
        return int(raw)
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_metadata(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {"fonts": []}
    current_font: Optional[Dict[str, Any]] = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("fonts {"):
            current_font = {}
            continue

        if stripped == "}":
            if current_font is not None:
                data["fonts"].append(current_font)
                current_font = None
            continue

        if ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = parse_value(raw_value)

        target = current_font if current_font is not None else data

        if key in target:
            existing = target[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                target[key] = [existing, value]
        else:
            if key in {"subsets", "designers", "languages", "axes", "classifications"}:
                target[key] = [value]
            else:
                target[key] = value

    return data


def iter_metadata_files(root: Path, licenses: Iterable[str]) -> Iterable[Path]:
    for license_dir in licenses:
        base = root / license_dir
        if not base.exists():
            continue
        yield from base.rglob("METADATA.pb")


def style_key_from_entry(entry: Dict[str, Any]) -> str:
    weight = entry.get("weight", 400)
    style = entry.get("style", "normal").lower()
    if style in {"normal", "regular"}:
        suffix = ""
    elif style == "italic":
        suffix = "italic"
    else:
        suffix = style
    return f"{weight}{suffix}"


def human_style_name(entry: Dict[str, Any]) -> str:
    weight = entry.get("weight", 400)
    style = entry.get("style", "normal").lower()
    weight_names = {
        100: "Thin",
        200: "ExtraLight",
        300: "Light",
        400: "Regular",
        500: "Medium",
        600: "SemiBold",
        700: "Bold",
        800: "ExtraBold",
        900: "Black",
    }
    base = weight_names.get(weight, str(weight))
    if style == "italic":
        return f"{base} Italic"
    if style not in {"normal", "regular"}:
        return f"{base} {style.title()}"
    return base


def build_summary(
    metadata_files: Iterable[Path],
    styles_filter: Optional[set[str]],
    base_dir: Path,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for meta_path in tqdm(list(metadata_files), desc="Indexing fonts"):
        record = parse_metadata(meta_path)
        family = record.get("name") or record.get("family")
        if not family:
            continue
        category = record.get("category", "unknown")
        designers = record.get("designer") or record.get("designers") or []
        if isinstance(designers, list):
            designer = ", ".join(designers)
        else:
            designer = designers
        subsets = record.get("subsets", [])
        license_dir = meta_path.parent

        for font_entry in record.get("fonts", []):
            style_key = style_key_from_entry(font_entry).lower()
            if styles_filter and style_key not in styles_filter:
                continue

            filename = font_entry.get("filename")
            if not filename:
                continue
            font_path = (license_dir / filename).resolve()
            if not font_path.exists():
                continue

            results.append(
                {
                    "family": family,
                    "style": human_style_name(font_entry),
                    "style_key": style_key,
                    "path": str(font_path),
                    "category": category,
                    "designer": designer,
                    "subsets": subsets,
                    "license": license_dir.name,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local Google Fonts repository.")
    parser.add_argument("--fonts-root", type=Path, required=True, help="Path to google/fonts repository.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store metadata summary.")
    parser.add_argument(
        "--styles",
        nargs="+",
        default=None,
        help="Optional style keys filter (e.g. regular, italic, 700, 700italic). If omitted, all available styles are kept.",
    )
    parser.add_argument(
        "--licenses",
        nargs="+",
        default=["ofl", "apache", "ufl"],
        help="Top-level directories (licenses) to scan inside fonts repository.",
    )
    args = parser.parse_args()

    styles_filter = {style.lower() for style in args.styles} if args.styles else None

    metadata_files = list(iter_metadata_files(args.fonts_root, args.licenses))
    summary = build_summary(metadata_files, styles_filter, args.fonts_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "fonts_index.json"

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"Indexed {len(summary)} font styles")
    print(f"Metadata saved to {summary_path}")


if __name__ == "__main__":
    main()
