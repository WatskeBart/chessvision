"""Export a trained YOLO checkpoint for fast inference on the RPi4B.

The Pi is CPU-only (no usable GPU for YOLO), so the format and input size matter
a lot. NCNN is the fastest backend on ARM CPUs; a smaller --imgsz trades a little
accuracy for a large speedup. Export a couple of sizes and benchmark on the Pi.

Usage:
    uv run export_model.py                          # ncnn, imgsz 640
    uv run export_model.py --imgsz 416              # smaller/faster
    uv run export_model.py --format onnx --imgsz 320
"""

import argparse
from pathlib import Path

from settings import settings


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--weights", type=Path, default=Path("models/pieces.pt"),
        help="Trained .pt checkpoint to export.",
    )
    p.add_argument(
        "--format", default="ncnn", choices=["ncnn", "onnx"],
        help="ncnn = fastest on ARM CPU (recommended for the Pi).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference input size. Smaller = faster on the Pi (try 416 or 320).",
    )
    p.add_argument(
        "--half", action="store_true",
        help="FP16 export (onnx only; ignored for ncnn CPU).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    if not args.weights.exists():
        raise SystemExit(f"ERROR: weights not found: {args.weights.resolve()}")

    print(f"[export] {args.weights}  →  format={args.format}  imgsz={args.imgsz}")
    model = YOLO(str(args.weights), task="detect")
    path = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"[export] wrote: {path}")
    print(
        "\nTo use it on the Pi, point detect_pieces.py at the export, e.g.:\n"
        f"  GRANDMASTER_PIECES_MODEL_PATH={path} uv run detect_pieces.py\n"
        f"(also matches settings.pieces_model_path → {settings.pieces_model_path})"
    )


if __name__ == "__main__":
    main()
