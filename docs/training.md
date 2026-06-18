# Training & datasets

Tooling to build a dataset of *your own* pieces, fine-tune the YOLO detector, and
export it for fast inference. All commands run from the repo root.

## Capturing a dataset (`gm-capture`)

Builds a YOLO-format dataset from the live stream without manual annotation. Set
up a physical position, run the command with the matching FEN, and press `space`
to save frames. The tool uses the current model for bounding-box proposals but
replaces each box's class with the ground-truth piece from the FEN. Squares the
model missed get a synthesized grid-cell box so the label set stays complete.

```bash
uv run gm-capture \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" \
    --infer-every 5   # run inference every 5th frame for a faster display
```

| Key | Action |
| --- | --- |
| `space` / `s` | Save current frame + label file |
| `e` | Type a new FEN in the terminal (no restart needed) |
| `v` | Paste FEN from clipboard (e.g. copied from lichess.org/editor) |
| `m` | Type a UCI move (e.g. `a7a6`) to update the position incrementally |
| `f` | Toggle board flip orientation |
| `r` | Rotate board view 90° clockwise (cycles 0→90→180→270) |
| `n` | Skip to next frame |
| `ESC` / `q` | Quit |

Every 5th capture (configurable with `--val-every`) goes to the `valid/` split;
the rest go to `train/`. A `data.yaml` is written once at startup.

> **Tip:** Use [lichess.org/editor](https://lichess.org/editor) to build positions
> fast. Drag pieces onto the board, then copy the FEN (it updates live in the URL
> and the FEN box). Press `v` in `gm-capture` to paste it straight from the
> clipboard — no retyping. To cover many positions quickly, set up one
> arrangement on lichess, paste it with `v`, capture a few frames, then tweak the
> board and repeat. This sweeps through a wide variety of layouts in minutes.

## Batch auto-labeling (`gm-autolabel`)

Same idea as `gm-capture` but for a folder of images you have already saved (e.g.
snapshots from `gm-detect`). All images must show the same known position. Pass
`--warped` if the images are already top-down warped boards.

```bash
uv run gm-autolabel --fen "<placement>" --src samples/
uv run gm-autolabel --fen "<placement>" --src warped/ --warped
```

## Training (`gm-train`)

[`chessvision/training/train.py`](../chessvision/training/train.py) fine-tunes a
YOLO checkpoint on the chess-pieces dataset, validates on the val split, and saves
the best weights under `models/`. Point it at your captured dataset and the
existing weights:

```bash
GRANDMASTER_TRAIN_DATA_YAML=datasets/my-pieces/data.yaml \
GRANDMASTER_TRAIN_MODEL_PATH=models/pieces.pt \
uv run gm-train
```

See [configuration](configuration.md#training-settings) for the training knobs.

## Export (`gm-export`)

[`chessvision/training/export.py`](../chessvision/training/export.py) converts a
trained `.pt` checkpoint to NCNN or ONNX for fast CPU inference on the Raspberry
Pi. NCNN is the recommended format for ARM.

```bash
uv run gm-export                    # ncnn, imgsz 640 (default)
uv run gm-export --imgsz 416        # smaller and faster
uv run gm-export --format onnx --imgsz 320
```

To use an exported model, point `GRANDMASTER_PIECES_MODEL_PATH` at the result.
ONNX is a single file (`models/pieces.onnx`); **NCNN is a directory** —
`gm-export` writes `models/pieces_ncnn_model/` containing `model.ncnn.param`,
`model.ncnn.bin` and `metadata.yaml`. Set the env var to the directory itself,
not a file inside it:

```bash
GRANDMASTER_PIECES_MODEL_PATH=models/pieces_ncnn_model uv run gm-detect
```

The `ncnn` runtime is a project dependency (installed by `uv sync`), so NCNN
models load without any extra setup.
