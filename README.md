# chessvision aka Grandmaster Yolo

Track a live chess game from a single overhead camera. A phone streams the board
over MJPEG, the pipeline finds the board, perspective-warps it to a top-down
view, and a YOLO model fine-tuned on chess pieces reports what sits on each
square.

## How it works

The pipeline runs per frame in [detect_pieces.py](detect_pieces.py). Board
detection and warping live in [detect_chessboard.py](detect_chessboard.py),
which [detect_pieces.py](detect_pieces.py) imports:

1. **Find the board** — `find_board_corners()` detects the rank/file grid lines
   with a Hough transform and takes the outermost line on each side as the board
   boundary. This is robust to a broken or partially occluded outer border. If
   the grid lines can't be resolved (low contrast, too few squares visible) it
   falls back to finding the largest 4-sided contour.
2. **Warp** — `warp_board()` perspective-warps the frame to a square top-down
   view of the board.
3. **Detect pieces** — a YOLO model (12 classes: 6 piece types × {white, black})
   runs on the warped view.
4. **Map to squares** — `detections_to_board()` assigns each detection to a
   square using the box's bottom-center point (a piece's base sits on its
   square), keeping the highest-confidence detection per square, and prints the
   occupied squares as `e1:white-king, ...`.

## Setup

Requires Python ≥ 3.13. The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Configuration

Settings are defined in [settings.py](settings.py) and read from a `.env` file
(all keys are prefixed `GRANDMASTER_`). Copy the example and edit it:

```bash
cp .env.example .env
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `GRANDMASTER_STREAM_URL` | `http://10.42.0.177:4444/stream` | Camera stream URL to read frames from |
| `GRANDMASTER_PIECES_MODEL_PATH` | `runs/detect/train/weights/best.pt` | YOLO model fine-tuned on chess pieces (inference) |
| `GRANDMASTER_TRAIN_MODEL_PATH` | `yolo26n.pt` | Base checkpoint to fine-tune from |
| `GRANDMASTER_TRAIN_DATA_YAML` | `datasets/chess-pieces/data.yaml` | Training dataset (YOLO format) |
| `GRANDMASTER_TRAIN_EPOCHS` | `100` | Training epochs |
| `GRANDMASTER_TRAIN_IMGSZ` | `640` | Training image size |
| `GRANDMASTER_TRAIN_PATIENCE` | `20` | Early-stopping patience |
| `GRANDMASTER_TRAIN_DEVICE` | `0` | GPU index for training |
| `GRANDMASTER_FLIP_ORIENTATION` | `true` | Set `true` when the warped top-left square is a1; leave `false` when it's a8 |
| `GRANDMASTER_DISPLAY_SIZE` | `800` | Minimum side length (px) to upscale the displayed board to |

## Usage

```bash
# Preview the raw camera stream
uv run view_camera.py

# Debug board detection (corner overlay + warp), no piece model needed
uv run detect_chessboard.py

# Full pipeline: detect the board and the pieces on it
uv run detect_pieces.py

# Fine-tune the piece detection model
uv run train_pieces.py
```

Press `ESC` to close any of the OpenCV preview windows.

### Training

[train_pieces.py](train_pieces.py) fine-tunes the COCO-pretrained nano
checkpoint on the chess-pieces dataset, validates on the val split, and exports
the best weights to ONNX for fast inference on the Raspberry Pi (swap to
`format="ncnn"` for even faster ARM CPU inference). Training artifacts land
under `runs/detect/`.

## Hardware setup

```mermaid
flowchart LR
    Internet(("🌐 Internet"))
    Laptop["💻 Laptop<br/>shares its Wi-Fi Internet<br/>connection (acts as router)"]

    subgraph netA["Network A — RPi Wi-Fi hotspot"]
        direction LR
        Phone["📱 Android phone<br/>IP camera app<br/>(on tripod above board)"]
        RPi["🖥️ Raspberry Pi 4B<br/>hosts the hotspot<br/>runs the vision pipeline"]
        Phone -- "Wi-Fi" --> RPi
    end

    Internet -- "Wi-Fi" --> Laptop
    Laptop -- "Ethernet (Internet sharing)" --> RPi
```

| Component | Role |
| --- | --- |
| Android phone | Mounted on a tripod above the board, streams video via an IP camera app |
| Raspberry Pi 4B | Hosts its own Wi-Fi hotspot for the phone to join; runs the vision pipeline (`detect_pieces.py`) |
| Laptop | Connected to the RPi over Ethernet; shares its own Wi-Fi Internet connection with the RPi (acts as a router) |
| Internet | Reached by the laptop over Wi-Fi, then passed through to the RPi via Ethernet |

## Proof of Concepts

### POC 1

Detect the chessboard

### POC 2

Detect individual pieces and track them
