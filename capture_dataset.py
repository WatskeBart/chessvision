"""Model-assisted auto-labeling capture tool.

Captures warped top-down board frames from the live stream and writes
YOLO-format training labels for *your own* pieces, without manual annotation.

The trick: you physically set up a position you specify as a FEN, so you know
exactly which piece sits on every square. We then run the *current* model on the
warped frame and keep its predicted boxes (localization transfers across piece
appearance), but **override each box's class** with the ground-truth piece and
**drop boxes on empty squares**. Squares that hold a known piece the model
missed get a synthesized grid-cell box so the label set stays complete.

Capture many frames of each position under varied lighting/angles, change the
physical position, and rerun. Fine-tune from models/pieces.pt on the result.

Usage:
    uv run capture_dataset.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

Keys (in the window):
    space / s  capture the current frame (image + label, into the target split)
    e          enter a new FEN in the terminal (no restart needed)
    f          toggle board flip orientation (fix mirrored square labels)
    r          rotate board view 90° clockwise (cycles 0→90→180→270→0)
    n          force a fresh stream read (skip ahead)
    ESC / q    quit
"""

import argparse
import time
from pathlib import Path

import chess
import cv2
import numpy as np

from detect_corners_cv import board_quad_from_corners, find_corners, warp_board
from detect_pieces import scale_for_display
from settings import settings

# Canonical class order — MUST match datasets/chess-pieces/data.yaml so class
# IDs stay aligned when fine-tuning from the existing models/pieces.pt weights.
CLASS_NAMES = [
    "black-bishop", "black-king", "black-knight", "black-pawn",
    "black-queen", "black-rook", "white-bishop", "white-king",
    "white-knight", "white-pawn", "white-queen", "white-rook",
]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# FEN piece letter -> class name. Uppercase = white, lowercase = black.
_PIECE = {
    "p": "pawn", "n": "knight", "b": "bishop",
    "r": "rook", "q": "queen", "k": "king",
}
FEN_TO_CLASS = {
    sym: f"{'white' if sym.isupper() else 'black'}-{_PIECE[sym.lower()]}"
    for sym in "PNBRQKpnbrqk"
}

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 first → top-left of the warped board

_CV2_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def short_label(cls_name):
    """'white-knight' -> 'wN', for compact on-screen text."""
    color, piece = cls_name.split("-")
    initial = {"knight": "N"}.get(piece, piece[0].upper())
    return f"{color[0]}{initial}"


_TEXT_CCW_FLAGS = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


def _put_text_rotated(img, text, org, font, scale, color, thickness, ccw_deg):
    """Draw text at org (bottom-left anchor), rotated ccw_deg° counter-clockwise."""
    if ccw_deg == 0:
        cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)
        return
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 1
    tmp = np.zeros((th + baseline + 2 * pad, tw + 2 * pad, 3), dtype=np.uint8)
    cv2.putText(tmp, text, (pad, th + pad), font, scale, color, thickness, cv2.LINE_AA)
    tmp = cv2.rotate(tmp, _TEXT_CCW_FLAGS[ccw_deg])
    rh, rw = tmp.shape[:2]
    tx, ty = x, y - rh
    ix1 = max(0, tx); ix2 = min(img.shape[1], tx + rw)
    iy1 = max(0, ty); iy2 = min(img.shape[0], ty + rh)
    if ix1 >= ix2 or iy1 >= iy2:
        return
    patch = tmp[iy1 - ty:iy2 - ty, ix1 - tx:ix2 - tx]
    mask = patch.any(axis=2)
    img[iy1:iy2, ix1:ix2][mask] = patch[mask]


def grid_to_square(col, row, flip, rotation=0):
    if rotation == 90:
        col, row = row, 7 - col
    elif rotation == 180:
        col, row = 7 - col, 7 - row
    elif rotation == 270:
        col, row = 7 - row, col
    if flip:
        col, row = 7 - col, 7 - row
    return f"{FILES[col]}{RANKS[row]}"


def square_to_grid(sq, flip, rotation=0):
    col, row = FILES.index(sq[0]), RANKS.index(sq[1])
    if flip:
        col, row = 7 - col, 7 - row
    # Inverse rotation
    if rotation == 90:
        col, row = 7 - row, col
    elif rotation == 180:
        col, row = 7 - col, 7 - row
    elif rotation == 270:
        col, row = row, 7 - col
    return col, row


def fen_to_ground_truth(fen):
    """Parse a FEN (full or piece-placement field only) into {square: class}."""
    board = chess.Board(None)  # empty board
    board.set_board_fen(fen.split()[0])  # validates and parses placement
    gt = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            gt[chess.square_name(square)] = FEN_TO_CLASS[piece.symbol()]
    return gt


def assign_square(x1, y1, x2, y2, board_w, board_h, padding, flip, rotation=0):
    """Map a box to a square using the box's bottom-center point — identical to
    the inference-time mapping in detect_pieces.detections_to_board."""
    cell_w = (board_w - 2 * padding) / 8
    cell_h = (board_h - 2 * padding) / 8
    cx = (x1 + x2) / 2 - padding
    cy = y2 - padding
    col = min(7, max(0, int(cx // cell_w)))
    row = min(7, max(0, int(cy // cell_h)))
    return grid_to_square(col, row, flip, rotation)


def build_labels(result, gt, board_w, board_h, padding, flip, synthesize, rotation=0):
    """Return (labels, stats). labels: list of (cls_id, xc, yc, w, h, source) in
    normalized [0,1] coords. stats: dict of matched/missing/dropped counts."""
    # Keep the highest-confidence model box per occupied square.
    best = {}  # square -> (conf, (x1,y1,x2,y2), cls_name)
    dropped = 0
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        sq = assign_square(x1, y1, x2, y2, board_w, board_h, padding, flip, rotation)
        if sq not in gt:
            dropped += 1  # detection on an empty square -> false positive, drop
            continue
        if sq not in best or conf > best[sq][0]:
            best[sq] = (conf, (x1, y1, x2, y2), gt[sq])

    labels = []
    box_sizes = []
    for _, (_, (x1, y1, x2, y2), cls_name) in best.items():
        labels.append(_norm_box(x1, y1, x2, y2, board_w, board_h, cls_name, "model"))
        box_sizes.append((x2 - x1, y2 - y1))

    missing = sorted(set(gt) - set(best))
    if synthesize and missing:
        # Size synthesized boxes from the median detected box this frame, so they
        # match the apparent piece scale; fall back to ~0.8 of a cell if none.
        cell_w = (board_w - 2 * padding) / 8
        cell_h = (board_h - 2 * padding) / 8
        if box_sizes:
            bw = sorted(w for w, _ in box_sizes)[len(box_sizes) // 2]
            bh = sorted(h for _, h in box_sizes)[len(box_sizes) // 2]
        else:
            bw, bh = 0.8 * cell_w, 0.8 * cell_h
        for sq in missing:
            col, row = square_to_grid(sq, flip, rotation)
            ccx = padding + (col + 0.5) * cell_w
            ccy = padding + (row + 0.5) * cell_h
            labels.append(
                _norm_box(
                    ccx - bw / 2, ccy - bh / 2, ccx + bw / 2, ccy + bh / 2,
                    board_w, board_h, gt[sq], "synth",
                )
            )

    stats = {
        "matched": len(best),
        "missing": len(missing),
        "synth": len(missing) if synthesize else 0,
        "dropped": dropped,
        "expected": len(gt),
    }
    return labels, stats


def _norm_box(x1, y1, x2, y2, w, h, cls_name, source):
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    clip = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    return (CLASS_ID[cls_name], clip(xc), clip(yc), clip(bw), clip(bh), source)


def draw_overlay(warped, labels, rotation):
    """Annotate the warped board with detection boxes and piece labels."""
    out = warped.copy()
    h, w = out.shape[:2]
    for cls_id, xc, yc, bw, bh, source in labels:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = (0, 255, 0) if source == "model" else (255, 200, 0)  # green / cyan
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        _put_text_rotated(
            out, short_label(CLASS_NAMES[cls_id]), (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, rotation,
        )
    return out


def draw_header(img, stats, flip, rotation, split, captured):
    """Draw the status banner at the top of img; always returns a fresh copy."""
    out = img.copy()
    w = out.shape[1]
    header = (
        f"flip:{'ON' if flip else 'OFF'}  rot:{rotation}°  split:{split}  saved:{captured}  |  "
        f"matched {stats['matched']}/{stats['expected']}  "
        f"synth {stats['synth']}  dropped {stats['dropped']}"
    )
    cv2.rectangle(out, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(
        out, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        cv2.LINE_AA,
    )
    return out


def write_capture(out_dir, split, stem, warped, labels, save_viz, viz_img):
    img_dir = out_dir / split / "images"
    lbl_dir = out_dir / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_dir / f"{stem}.jpg"), warped)
    lines = [
        f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        for c, xc, yc, bw, bh, _ in labels
    ]
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    if save_viz:
        viz_dir = out_dir / split / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(viz_dir / f"{stem}.jpg"), viz_img)


def write_data_yaml(out_dir):
    """Write a data.yaml pointing at the captured train/valid splits."""
    names = "\n".join(f"  - {n}" for n in CLASS_NAMES)
    content = (
        f"path: {out_dir.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names}\n"
    )
    (out_dir / "data.yaml").write_text(content)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--fen", default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        help="Position currently on the board (FEN piece-placement field). "
        "Default: standard starting position.",
    )
    p.add_argument("--out", type=Path, default=Path("datasets/my-pieces"))
    p.add_argument(
        "--model", type=Path, default=settings.pieces_model_path,
        help="Model to propose boxes (pieces.pt gives better proposals than onnx).",
    )
    p.add_argument(
        "--val-every", type=int, default=5,
        help="Send every Nth capture to the valid split (default 5 = ~20%% val).",
    )
    p.add_argument(
        "--propose-conf", type=float, default=0.20,
        help="Low conf for box proposals; empty-square boxes are dropped anyway.",
    )
    p.add_argument(
        "--no-synthesize", action="store_true",
        help="Do not add grid-cell boxes for known pieces the model missed.",
    )
    p.add_argument("--viz", action="store_true", help="Also save annotated viz images.")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    gt = fen_to_ground_truth(args.fen)
    print(f"[init] ground-truth: {len(gt)} pieces from FEN")

    if not args.model.exists():
        raise SystemExit(f"ERROR: model not found: {args.model.resolve()}")
    print(f"[init] loading model: {args.model}")
    model = YOLO(str(args.model), task="detect")

    args.out.mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.out)
    print(f"[init] dataset → {args.out.resolve()} (data.yaml written)")

    url = str(settings.stream_url)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open stream: {url}")
    print(f"[init] stream opened: {url}")

    flip = settings.flip_orientation
    rotation = 0
    synthesize = not args.no_synthesize
    session = time.strftime("%Y%m%d_%H%M%S")
    captured = 0

    print("\nKeys: [space/s] capture  [e] new FEN  [f] flip  [r] rotate  [n] skip  [ESC/q] quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[stream] read failed, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(url)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray)

        if corners is None:
            view = frame.copy()
            cv2.putText(
                view, "no board detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2,
            )
            cv2.imshow(
                "Capture dataset", scale_for_display(view, settings.display_size)
            )
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
            continue

        quad = board_quad_from_corners(corners)
        try:
            warped = warp_board(frame, quad, padding=settings.warp_padding)
        except cv2.error:
            continue

        result = model(warped, verbose=False, conf=args.propose_conf)[0]
        bh, bw = warped.shape[:2]
        labels, stats = build_labels(
            result, gt, bw, bh, settings.warp_padding, flip, synthesize, rotation
        )

        to_valid = captured % args.val_every == args.val_every - 1
        split = "valid" if to_valid else "train"
        viz = draw_overlay(warped, labels, rotation)
        display = cv2.rotate(viz, _CV2_ROTATIONS[rotation]) if rotation else viz
        display = draw_header(display, stats, flip, rotation, split, captured)
        cv2.imshow("Capture dataset", scale_for_display(display, settings.display_size))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("e"):
            new_fen = input("[FEN] Enter new FEN: ").strip()
            if new_fen:
                try:
                    gt = fen_to_ground_truth(new_fen)
                    print(f"[FEN] updated — {len(gt)} pieces")
                except Exception as exc:
                    print(f"[FEN] invalid FEN, keeping previous ({exc})")
        elif key == ord("f"):
            flip = not flip
            print(f"[toggle] flip → {'ON' if flip else 'OFF'} "
                  "(verify square labels match the real board)")
        elif key == ord("r"):
            rotation = (rotation + 90) % 360
            print(f"[toggle] rotation → {rotation}°")
        elif key == ord("n"):
            continue
        elif key in (ord(" "), ord("s")):
            stem = f"{session}_{captured:04d}"
            write_capture(args.out, split, stem, warped, labels, args.viz, viz)
            captured += 1
            print(
                f"[capture {captured}] {split}/{stem}  matched {stats['matched']}/"
                f"{stats['expected']}  synth {stats['synth']}  "
                f"dropped {stats['dropped']}"
            )
            if stats["dropped"] > 2:
                print("  ! several boxes dropped on empty squares — check FEN/flip "
                      "or raise --propose-conf")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[done] captured {captured} frames into {args.out.resolve()}")
    print("Next: fine-tune from models/pieces.pt, e.g.")
    print(f"  GRANDMASTER_TRAIN_DATA_YAML={args.out}/data.yaml \\")
    print("  GRANDMASTER_TRAIN_MODEL_PATH=models/pieces.pt uv run train_pieces.py")


if __name__ == "__main__":
    main()
