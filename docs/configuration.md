# Configuration

All settings are defined in
[`chessvision/settings.py`](../chessvision/settings.py) with sensible defaults,
so the app runs without any configuration. To override a value, set the matching
environment variable (prefix `GRANDMASTER_`), either in your shell or in a `.env`
file at the repo root:

```bash
cp .env.example .env   # then edit
```

Values are read from the environment at startup; run commands from the repo root
so the relative paths (`models/`, `datasets/`, `games/`) resolve.

## Settings

| Setting (env var) | Default | Purpose |
| --- | --- | --- |
| `GRANDMASTER_STREAM_URL` | `http://10.42.0.177:4444/stream` | Camera/MJPEG stream URL to read frames from |
| `GRANDMASTER_PIECES_MODEL_PATH` | `models/pieces_ncnn_model` | YOLO model for inference. A single-file weight (`.pt`/`.onnx`) **or** an NCNN export, which is a *directory* — point at the folder, not a file inside it |
| `GRANDMASTER_DETECTION_MODE` | `diff` | `diff` (model-free image subtraction) or `model` (YOLO every frame). Toggle live with `m` |
| `GRANDMASTER_SHOW_DETECTIONS` | `false` | Draw the detection overlay (boxes / changed-square outlines) at startup. Detection still runs when off; toggle with `d` |
| `GRANDMASTER_PIECES_CONF_THRESHOLD` | `0.3` | Minimum YOLO confidence to accept a detection (`model` mode / `v` check); raise (0.5–0.7) to suppress false positives |
| `GRANDMASTER_BOARD_ROTATION` | `90` | Clockwise rotation (0/90/180/270) of the square-name mapping — see [Board orientation](#board-orientation). **Cannot be set via env** (see note); change in `settings.py` or with `o` |
| `GRANDMASTER_FLIP_ORIENTATION` | `false` | Extra 180° turn of the square mapping, on top of `board_rotation`. Toggle with `f` |
| `GRANDMASTER_DISPLAY_SIZE` | `800` | Minimum side length (px) to upscale the displayed board to |
| `GRANDMASTER_WARP_PADDING` | `40` | Extra px of context kept around the warped board so overhanging pieces stay visible |
| `GRANDMASTER_GAME_STABILITY_FRAMES` | `6` | Consecutive identical frames before a board state is accepted and a move recorded |
| `GRANDMASTER_GAMES_DIR` | `games` | Directory where recorded games (`.pgn` + `.fen.log`) are written |
| `GRANDMASTER_GAME_DEBUG` | `false` | Verbose recording diagnostics (stability progress, match scores) |
| `GRANDMASTER_DIFF_CHANGE_THRESHOLD` | `10.0` | `diff` mode: mean grayscale change (0–255) for a square to count as "changed". Lower if low-contrast moves are missed; raise to ignore shadows |
| `GRANDMASTER_DIFF_INNER_FRACTION` | `0.7` | `diff` mode: central fraction of each square sampled (avoids grid lines / leaning pieces) |
| `GRANDMASTER_DIFF_MAX_CHANGED` | `6` | `diff` mode: frames with more changed squares than this are treated as noise (e.g. a hand) and skipped |
| `GRANDMASTER_DIFF_TOLERANCE` | `1` | `diff` mode: extra changed squares tolerated beyond a move's own squares when matching |

### Training settings

Only used by `gm-train` / `gm-capture` (see [training](training.md)):

| Setting (env var) | Default | Purpose |
| --- | --- | --- |
| `GRANDMASTER_TRAIN_MODEL_PATH` | `models/yolo26n.pt` | Base checkpoint to fine-tune from |
| `GRANDMASTER_TRAIN_DATA_YAML` | `datasets/combined/data.yaml` | Training dataset (YOLO format). The combined dataset merges the external `chess-pieces` dataset with your own captured `my-pieces` images |
| `GRANDMASTER_TRAIN_EPOCHS` | `40` | Training epochs |
| `GRANDMASTER_TRAIN_IMGSZ` | `640` | Training image size |
| `GRANDMASTER_TRAIN_PATIENCE` | `20` | Early-stopping patience |
| `GRANDMASTER_TRAIN_DEVICE` | `0` | GPU index, or `cpu` |

## Board orientation

The on-screen view is always left in the **raw camera-feed orientation**.
`board_rotation` and `flip_orientation` rotate only the *square-name mapping* (a
coordinate transform), so the board reads correctly without rotating the picture.

- `board_rotation` (0/90/180/270, clockwise) corrects a camera mounted a
  quarter/three-quarter turn off. The shipped default `90` matches a camera whose
  raw feed shows **a1 in the bottom-right** (mapping makes bottom-right read as a1
  and top-left as h8). The on-screen `a1` / `h8` corner markers confirm it.
- `flip_orientation` adds the remaining 180°.
- Tune live with `o` (rotation) and `f` (flip); the markers update immediately.

> **Note:** `GRANDMASTER_BOARD_ROTATION` can't be set via the environment — the
> field is typed `Literal[0, 90, 180, 270]` and pydantic rejects the env *string*
> `"90"`. Change the default in `settings.py`, or just cycle it at runtime with
> `o`. (`flip_orientation` is a bool and coerces from env fine.)
