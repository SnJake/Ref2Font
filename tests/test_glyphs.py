import unittest

import numpy as np
from PIL import Image
from fontTools.pens.ttGlyphPen import TTGlyphPen

from flux_grid_to_ttf import clean_cell_components, contours_from_image, draw_glyph_fixed_grid
from glyph_metrics import normalize_metrics, bounds


def rectangle(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def entry(ch, contours, row=0):
    return dict(char=ch, contours=contours, row=row, baseline_px=50.0,
                y_shift_px=0.0, visual_center_x=None)


class GlyphTests(unittest.TestCase):
    def test_no_reference_letters_retains_existing_placement(self):
        glyph = entry("p", [rectangle(0, 20, 20, 50)])
        self.assertEqual(normalize_metrics([glyph]), (None, None))
        self.assertEqual(bounds(glyph["contours"])[1], -30)
        self.assertEqual(bounds(glyph["contours"])[3], 20)

    def test_empty_core_does_not_borrow_a_neighbor(self):
        pixels = np.zeros((20, 20), dtype=np.uint8)
        pixels[2:18, :3] = 255
        pixels[10, 10] = 255
        cleaned = clean_cell_components(Image.fromarray(pixels), False, 127,
                                        3, 6, 0.25, 0.35, (5, 5, 15, 15))
        self.assertFalse(np.array(cleaned).any())

    def test_cleanup_keeps_antialiasing_but_removes_noise(self):
        pixels = np.zeros((20, 20), dtype=np.uint8)
        pixels[5:15, 6:14] = 255
        pixels[5:15, 5] = 80
        pixels[1, 1] = 255
        for invert in (False, True):
            image = Image.fromarray(255 - pixels if invert else pixels)
            cleaned = np.array(clean_cell_components(image, invert, 127, 1, 3, 0, 0.25))
            if invert:
                cleaned = 255 - cleaned
            self.assertEqual(cleaned[10, 5], 80)
            self.assertEqual(cleaned[1, 1], 0)

    def test_edge_touching_contours_are_closed_and_keep_counter(self):
        pixels = np.full((24, 24), 255, dtype=np.uint8)
        pixels[7:17, 7:17] = 0
        contours, bbox = contours_from_image(Image.fromarray(pixels), 0.5, 0.1, 4, 0, 0, False, 0)
        self.assertEqual(len(contours), 2)
        self.assertLess(bbox[0], 0)
        self.assertGreater(bbox[2], 23)
        areas = [np.sum(p[:, 0] * np.roll(p[:, 1], 1) - p[:, 1] * np.roll(p[:, 0], 1)) for p in contours]
        self.assertLess(areas[0] * areas[1], 0, "counter must have opposite winding")

    def test_normalization_preserves_accents_descenders_and_punctuation(self):
        entries = [entry("H", [rectangle(0, 10, 30, 70)]),
                   entry("E", [rectangle(0, 14, 30, 72)]),
                   entry("n", [rectangle(0, 30, 25, 50)]),
                   entry("a", [rectangle(0, 35, 25, 52)]),
                   entry("p", [rectangle(0, 30, 25, 70)]),
                   entry("ё", [rectangle(0, 33, 25, 51), rectangle(2, 18, 5, 5), rectangle(17, 18, 5, 5)]),
                   entry(".", [rectangle(0, 40, 6, 6)]),
                   entry(",", [rectangle(0, 40, 6, 15)])]
        cap, xheight = normalize_metrics(entries)
        for e in entries[:4]:
            _, top, _, bottom = bounds(e["contours"])
            self.assertAlmostEqual(bottom, 0)
            self.assertAlmostEqual(-top, cap if e["char"].isupper() else xheight)
        self.assertGreater(bounds(entries[4]["contours"])[3], 0)
        self.assertEqual(len(entries[5]["contours"]), 3)
        self.assertLess(bounds(entries[5]["contours"])[1], -xheight)
        self.assertEqual(bounds(entries[6]["contours"])[3], 0)
        self.assertGreater(bounds(entries[7]["contours"])[3], 0)

    def test_optical_alignment_does_not_make_negative_side_bearings(self):
        pen = TTGlyphPen(None)
        advance = draw_glyph_fixed_grid(pen, [rectangle(10, 10, 30, 60)], None,
                                        10, 40, 100, 80, 100,
                                        align_mode="visual", visual_center_x=100)
        glyph = pen.glyph()
        glyph.recalcBounds(None)
        self.assertGreaterEqual(glyph.xMin, 0)
        self.assertGreaterEqual(advance - glyph.xMax, 0)


if __name__ == "__main__":
    unittest.main()
