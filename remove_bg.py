#!/usr/bin/env python3
"""Remove the solid black background from OFF Сянка product renders.

Strategy: flood-fill connected dark regions starting from the image border,
so the black "OFF" logo inside the (brightly coloured) pouch is preserved.
A 1px erosion of the kept foreground trims the dark anti-aliased halo.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

NAMES = ["caramel", "banana", "vanilla", "cherry", "blueberry",
         "apple", "cinnamon", "cookie", "fruit", "strawberry"]
T = 72          # max-channel brightness below this counts as "dark"
ERODE = 1       # px to trim halo

for name in NAMES:
    path = f"images/{name}.png"
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]

    dark = arr.max(axis=2) < T  # candidate background pixels

    # Label connected dark regions; keep only those touching the border.
    labels, n = ndimage.label(dark)
    border = np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ])
    border_ids = set(np.unique(border)) - {0}
    background = np.isin(labels, list(border_ids))

    foreground = ~background
    if ERODE:
        foreground = ndimage.binary_erosion(foreground, iterations=ERODE)

    alpha = np.where(foreground, 255, 0).astype(np.uint8)
    out = np.dstack([arr, alpha])
    Image.fromarray(out, "RGBA").save(path)
    pct = 100 * background.mean()
    print(f"{name}: removed {pct:.1f}% as background, {n} dark regions")

print("done")
