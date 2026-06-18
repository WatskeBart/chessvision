"""Batch model-assisted auto-labeling for already-saved frames.

Same idea as capture_dataset.py, but for a folder of images you've already saved
(e.g. snapshots from detect_pieces.py) rather than the live stream. All images
must show the *same* known position, given as a FEN.

By default each image is treated as a raw camera frame: corners are detected and
the board is warped first. Pass --warped if the images are already top-down
warped boards (e.g. saved with detect_pieces' snapshot key).

Usage:
    uv run gm-autolabel --fen "<placement>" --src samples
"""

import argparse
from pathlib import Path

import cv2

from chessvision.core.board import board_quad_from_corners, find_corners, warp_board
from chessvision.settings import settings
from chessvision.training.capture_dataset import (
    build_labels,
    draw_overlay,
    fen_to_ground_truth,
    write_capture,
    write_data_yaml,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--fen", required=True, help="FEN placement field for ALL images in --src."
    )
    p.add_argument("--src", type=Path, required=True, help="Folder of input images.")
    p.add_argument("--out", type=Path, default=Path("datasets/my-pieces"))
    p.add_argument("--model", type=Path, default=settings.pieces_model_path)
    p.add_argument("--split", default="train", choices=["train", "valid"])
    p.add_argument("--propose-conf", type=float, default=0.20)
    p.add_argument("--no-synthesize", action="store_true")
    p.add_argument(
        "--flip", action="store_true", help="Override settings.flip_orientation."
    )
    p.add_argument(
        "--warped", action="store_true", help="Inputs are already warped boards."
    )
    p.add_argument("--viz", action="store_true", help="Also save annotated viz images.")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    if not args.model.exists():
        raise SystemExit(f"ERROR: model not found: {args.model.resolve()}")
    images = sorted(p for p in args.src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"ERROR: no images found in {args.src.resolve()}")

    gt = fen_to_ground_truth(args.fen)
    flip = args.flip or settings.flip_orientation
    synthesize = not args.no_synthesize
    model = YOLO(str(args.model), task="detect")

    args.out.mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.out)
    print(
        f"[init] {len(images)} images, {len(gt)} GT pieces, "
        f"flip={flip} → {args.out.resolve()}"
    )

    written, skipped = 0, 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[skip] unreadable: {img_path.name}")
            skipped += 1
            continue

        if args.warped:
            warped = frame
        else:
            corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if corners is None:
                print(f"[skip] no board: {img_path.name}")
                skipped += 1
                continue
            warped = warp_board(
                frame, board_quad_from_corners(corners), padding=settings.warp_padding
            )

        result = model(warped, verbose=False, conf=args.propose_conf)[0]
        bh, bw = warped.shape[:2]
        labels, stats = build_labels(
            result, gt, bw, bh, settings.warp_padding, flip, synthesize
        )

        stem = img_path.stem
        viz = (
            draw_overlay(warped, labels, stats, flip, args.split, written)
            if args.viz
            else None
        )
        write_capture(args.out, args.split, stem, warped, labels, args.viz, viz)
        written += 1
        print(
            f"[{written}] {img_path.name} → matched "
            f"{stats['matched']}/{stats['expected']}  "
            f"synth {stats['synth']}  dropped {stats['dropped']}"
        )

    print(
        f"\n[done] wrote {written}, skipped {skipped} → "
        f"{args.out.resolve()}/{args.split}"
    )


if __name__ == "__main__":
    main()
