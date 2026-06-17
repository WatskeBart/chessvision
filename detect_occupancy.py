"""Model-free piece-move detection by image subtraction.

Instead of identifying *what* piece sits on each square (the hard problem the
YOLO detector struggles with), this compares the current warped board against a
reference image of the board as of the last committed move and reports *which
squares changed*. Combined with the known game state and the rules of chess
(see GameTracker.update_changed), the changed-square set is enough to infer the
move — no trained model required.

It relies on a fixed camera / locked board transform: the reference image is
only meaningful while the warp stays stable, so use the board-lock ('k') in
detect_pieces.py before recording in this mode.
"""

import cv2
import numpy as np


def square_change_scores(warped, reference, padding=0, inner=0.7):
    """Per-square change score between a warped board frame and a reference.

    Returns an 8x8 float array (row 0 = top of the warp, matching the pixel
    layout used by detections_to_board) holding the mean absolute grayscale
    difference inside the central `inner` fraction of each square. Sampling only
    the centre avoids grid lines and tall neighbouring pieces leaning across a
    boundary, and tolerates small warp jitter."""
    g_now = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g_ref = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(g_now, g_ref)

    h, w = diff.shape[:2]
    cell_w = (w - 2 * padding) / 8
    cell_h = (h - 2 * padding) / 8
    margin = (1 - inner) / 2

    scores = np.zeros((8, 8), dtype=np.float32)
    for row in range(8):
        for col in range(8):
            x0 = int(padding + (col + margin) * cell_w)
            x1 = int(padding + (col + 1 - margin) * cell_w)
            y0 = int(padding + (row + margin) * cell_h)
            y1 = int(padding + (row + 1 - margin) * cell_h)
            patch = diff[y0:y1, x0:x1]
            scores[row, col] = float(patch.mean()) if patch.size else 0.0
    return scores


def changed_squares(scores, threshold):
    """(col, row) cells whose change score exceeds `threshold`."""
    rows, cols = np.where(scores > threshold)
    return {(int(c), int(r)) for r, c in zip(rows, cols)}
