"""Opt-in typographic alignment for generated Latin and Cyrillic atlases.

The atlas is a collection of individually positioned drawings, not a typeset
line. Infer shared heights from ordinary letters before positioning accents,
descenders and punctuation. These are heuristics, so legacy alignment remains
available for unusual alphabets and deliberately irregular designs.
"""

import numpy as np


CAPS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
LOWER = set("abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
ASCENDERS = set("bdfhkltб")
DESCENDERS = set("gjpqyДЦЩдруфцщQФf")
ACCENTED = set("ijЁЙёй")


def bounds(contours):
    points = np.vstack(contours)
    return (*points.min(axis=0), *points.max(axis=0))


def body_bounds(entry):
    contours = entry["contours"]
    if entry["char"] in ACCENTED:
        # The largest outer contour is the body, not a dot, accent or counter.
        def area(p):
            return abs(np.sum(p[:, 0] * np.roll(p[:, 1], 1)
                              - p[:, 1] * np.roll(p[:, 0], 1)))
        contours = [max(contours, key=area)]
    return bounds(contours)


def height_group(ch):
    if ch in CAPS or ch in ASCENDERS or ch.isdigit() or ch in "!?&ф":
        return "cap"
    if ch in LOWER:
        return "x"
    return None


def normalize_metrics(entries):
    """Mutate contours into a common Y-down coordinate system, baseline zero.

    Ordinary letters establish robust target heights. Descenders use the height
    of their row's peers instead of stretching their entire bbox onto the body.
    Uniform scaling preserves stroke proportions; dots and accents move with
    their letter. Unknown symbols keep their legacy baseline placement.
    """
    samples = {"cap": [], "x": []}
    rows = {}
    for entry in entries:
        if not entry["contours"]:
            continue
        ch = entry["char"]
        group = height_group(ch)
        if group and ch not in DESCENDERS:
            _, top, _, bottom = body_bounds(entry)
            height = bottom - top
            if height > 0:
                samples[group].append(height)
                rows.setdefault((entry["row"], group), []).append(height)
    if not any(samples.values()):
        # A symbols-only/custom charset has no evidence for typographic heights.
        for entry in entries:
            baseline = entry["baseline_px"] - entry["y_shift_px"]
            for contour in entry["contours"]:
                contour[:, 1] -= baseline
            entry["baseline_px"] = entry["y_shift_px"] = 0.0
        return None, None
    cap = float(np.median(samples["cap"])) if samples["cap"] else 1.0
    xheight = float(np.median(samples["x"])) if samples["x"] else cap * 0.72
    if not samples["cap"]:
        cap = xheight / 0.72
    targets = {"cap": cap, "x": xheight}
    for entry in entries:
        if not entry["contours"]:
            continue
        ch = entry["char"]
        _, top, _, bottom = body_bounds(entry)
        height = max(bottom - top, 1e-6)
        group = height_group(ch)
        factor = 1.0
        baseline = entry["baseline_px"] - entry["y_shift_px"]
        if group:
            target = targets[group]
            if ch in DESCENDERS:
                peers = rows.get((entry["row"], group), samples[group])
                body_height = float(np.median(peers)) if peers else target
                factor = float(np.clip(target / body_height, 0.8, 1.25))
                baseline = top + target / factor
            else:
                factor = float(np.clip(target / height, 0.8, 1.25))
                baseline = bottom
        elif ch == ".":
            baseline = bottom
        elif ch == ",":
            baseline = top + min(height, xheight * 0.18)
        elif ch in ":;":
            baseline = bottom if ch == ":" else top + xheight * 0.7
            if ch == ":":
                factor = float(np.clip(xheight * 0.7 / height, 0.8, 1.25))
        elif ch in "\"'":
            baseline = top + cap
        elif ch in "-–—":
            baseline = (top + bottom) * 0.5 + xheight * 0.45
        for contour in entry["contours"]:
            contour[:, 0] *= factor
            contour[:, 1] = (contour[:, 1] - baseline) * factor
        if entry["visual_center_x"] is not None:
            entry["visual_center_x"] *= factor
        entry["baseline_px"] = 0.0
        entry["y_shift_px"] = 0.0
    return cap, xheight
