# chessvision

Track a live chess game from a single overhead camera. A phone streams the board
over MJPEG; the pipeline finds the board, perspective-warps it to a top-down
view, works out the move that was played, and records the game to PGN.

## Quickstart

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install (creates the venv, deps, and the `gm-*` commands)
uv sync

# 2. Point at your camera stream (optional — defaults to the reference rig).
echo 'GRANDMASTER_STREAM_URL=http://<phone-ip>:<port>/stream' >> .env

# 3. Check the camera, then run the app
uv run gm-view        # raw stream preview — press ESC to close
uv run gm-detect      # the full pipeline
```

## Recording a game

Recording is most reliable when the board transform is **locked** first (a full
set of pieces hides the grid lines the detector needs):

1. Set up the **empty** board and confirm it's detected (`c` shows the corners).
2. Press **`k`** to lock the transform (a `BOARD LOCKED` marker appears).
3. Place the pieces in the starting position, then press **`r`** to record.
4. Play. Each accepted move is written live to `games/game_<timestamp>.pgn` and `.fen.log`.
5. Press **`r`** again (or `ESC`) to stop and finalize.

To record from a mid-game position, pass a FEN:

```bash
uv run gm-detect --from-fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3"
```

## Keyboard toggles

| Key | Action |
| --- | --- |
| `h` | Show / hide the help overlay |
| `d` | Toggle the detection overlay |
| `c` | Toggle the corner overlay on the raw frame |
| `k` | Lock / unlock the board transform |
| `m` | Switch detection mode (`diff` ⇄ `model`) |
| `v` | Validate the current board against the model |
| `p` | Toggle printing the board state to stdout |
| `o` | Rotate the square mapping 90° |
| `f` | Toggle board flip orientation |
| `r` | Start / stop recording the game |
| `s` | Save the current frame as `snapshot_<n>.png` |
| `ESC` | Quit |

## Commands

| Command | What it does |
| --- | --- |
| `gm-detect` | Live board + piece detection and game recording |
| `gm-view` | Preview the raw camera stream |
| `gm-corners` | Board-detection debug (corner overlay + warp) |
| `gm-capture` | Capture a labelled training dataset |
| `gm-autolabel` | Auto-label a folder of saved frames |
| `gm-train` | Fine-tune the piece-detection model |
| `gm-export` | Export trained weights to NCNN/ONNX |
