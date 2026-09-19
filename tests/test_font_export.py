"""Exercise the full converter against both supplied V3 atlases."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fontTools.ttLib import TTFont
from PIL import ImageFont

from scripts.atlas_to_ttf import PRESET_CHARSETS


ROOT = Path(__file__).resolve().parents[1]


class FontExportTests(unittest.TestCase):
    def test_example_fonts(self):
        for language, suffix, capitals, lowercase, accent in [
            ("latin", "L", "ABCEHIKLMNPRSTUVWXZ", "acemnorsuvwxz", "i"),
            ("cyrillic", "Cyrillic", "АБВГЕЖЗИКЛМНОПРСТУХЧШЪЫЬЭЮЯ", "авгежзиклмностхчшъыьэюя", "ё"),
        ]:
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([
                    sys.executable, str(ROOT / "flux_pipeline.py"),
                    "--input", str(ROOT / "Example" / "V3" / f"Example_3_Output_{suffix}.png"),
                    "--output-dir", directory, "--no-upscale", "--use-grid",
                    "--language", language, "--trace-scale", "2", "--align-mode", "visual",
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                path = next(Path(directory).glob("*.ttf"))
                with TTFont(path) as font:
                    cmap = font.getBestCmap()
                    self.assertEqual(set(cmap), {ord(c) for c in PRESET_CHARSETS[language] + " "})
                    glyf = font["glyf"]
                    for ch in PRESET_CHARSETS[language]:
                        glyph = glyf[cmap[ord(ch)]]
                        self.assertGreater(glyph.numberOfContours, 0, ch)
                        advance, lsb = font["hmtx"][cmap[ord(ch)]]
                        self.assertEqual(lsb, glyph.xMin, ch)
                        self.assertGreaterEqual(advance - glyph.xMax, -1, ch)
                        self.assertGreaterEqual(font["hhea"].ascent, glyph.yMax, ch)
                        self.assertLessEqual(font["hhea"].descent, glyph.yMin, ch)
                    for group in (capitals, lowercase):
                        glyphs = [glyf[cmap[ord(ch)]] for ch in group]
                        self.assertLessEqual(max(g.yMax for g in glyphs) - min(g.yMax for g in glyphs), 1)
                        self.assertTrue(all(g.yMin == 0 for g in glyphs))
                    self.assertGreaterEqual(glyf[cmap[ord(accent)]].numberOfContours, 2)
                    self.assertEqual(font["OS/2"].version, 4)
                    self.assertTrue(font["OS/2"].fsSelection & 128)
                    self.assertEqual(font["hhea"].ascent, font["OS/2"].sTypoAscender)
                rendered = ImageFont.truetype(str(path), 48).getmask(capitals)
                self.assertIsNotNone(rendered.getbbox())


if __name__ == "__main__":
    unittest.main()
