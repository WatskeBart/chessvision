from pathlib import Path
from typing import Literal
from pydantic import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRANDMASTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),
    )

    # Camera/MJPEG stream URL to read frames from.
    stream_url: AnyUrl = AnyUrl("http://10.42.0.177:4444/stream")

    # Path to a YOLO model fine-tuned on chess pieces (12 classes: 6 piece
    # types x {white, black}), used for inference in detect_pieces.py. Accepts
    # any format Ultralytics can load: a .pt/.onnx file, or an NCNN export, which
    # is a *directory* (models/pieces_ncnn_model/ holding model.ncnn.param +
    # model.ncnn.bin + metadata.yaml).
    pieces_model_path: Path = Path("models/pieces_ncnn_model")

    # Base checkpoint to fine-tune from in train_pieces.py. yolo26n.pt is the
    # COCO-pretrained nano model - much faster to converge than from scratch.
    train_model_path: Path = Path("models/yolo26n.pt")

    # data.yaml that came with the chess-pieces dataset (YOLO format).
    train_data_yaml: Path = Path("datasets/chess-pieces/data.yaml")

    # Training hyperparameters for train_pieces.py.
    train_epochs: int = 100
    train_imgsz: int = 640
    train_patience: int = 20  # stop early if val performance plateaus
    # GPU index (e.g. 0) if you have an NVIDIA/CUDA GPU, or "cpu" to train on CPU.
    train_device: int | Literal["cpu"] = 0

    # Clockwise rotation (0/90/180/270 degrees) applied when mapping board cells
    # to square names — as a coordinate rotation, NOT to the displayed image. The
    # on-screen view is left in the raw camera-feed orientation; only the square
    # identification is rotated, so the board reads correctly without rotating the
    # picture. Cycle at runtime with 'o'. flip_orientation adds the extra 180°.
    # Default 90: the raw feed shows a1 in the bottom-right, so a 90° mapping makes
    # the bottom-right cell read as a1 (and the top-left as h8).
    board_rotation: Literal[0, 90, 180, 270] = 90

    # Extra 180° turn of the square-name mapping, on top of board_rotation. Toggle
    # at runtime with 'f'. Leave False unless the board still reads 180° rotated
    # after setting board_rotation.
    # Default False: with board_rotation=90 the raw feed already reads correctly
    # (bottom-right = a1, top-left = h8).
    flip_orientation: bool = False

    # Minimum confidence for a piece detection to be accepted. Raise this to
    # suppress false positives on empty squares (e.g. 0.5–0.7).
    pieces_conf_threshold: float = 0.3

    # The warped board is only as big as the board appears in the raw camera
    # frame, which can be quite small. Scale the displayed view up to at least
    # this many pixels on a side so detected pieces/boxes are easy to see.
    display_size: int = 800

    # Extra pixels of original image context to include around each edge of the
    # warped board. Pieces that overhang the detected board boundary remain
    # visible so the piece detector can still find them.
    warp_padding: int = 40

    # Game recording (track_game.py, toggled with "r" in detect_pieces.py).
    # Number of consecutive identical frames before a detected board state is
    # accepted as stable and a move is inferred. Higher = more robust to
    # detection flicker, but adds a little lag after each physical move.
    game_stability_frames: int = 6

    # Directory where recorded games (a .pgn file and a .fen.log) are written.
    games_dir: Path = Path("games")

    # Verbose game-recording diagnostics: log stability progress and, for each
    # stable board, how well the current position and the best legal move match
    # the detection. Use this to find out why moves aren't being recorded.
    game_debug: bool = False

    # Detection method used by detect_pieces.py:
    #   "model" - run the YOLO piece detector every frame (identifies pieces).
    #   "diff"  - model-free image subtraction: detect which squares changed vs
    #             the last position and infer the move from the rules. Needs a
    #             known start position and a locked board ('k'); the model stays
    #             loaded only as a manual re-sync check ('v'). Toggle at runtime
    #             with 'm'.
    detection_mode: Literal["model", "diff"] = "diff"

    # Whether the detection overlay (piece boxes/labels in model mode, changed-
    # square outlines in diff mode) is drawn on the view at startup. Toggle at
    # runtime with 'd'. Default False: start with no overlay shown (detection
    # still runs; only its drawing is hidden).
    show_detections: bool = False

    # "diff" mode tuning ---------------------------------------------------
    # Mean absolute grayscale difference (0-255) within a square's centre, above
    # which the square counts as "changed" vs the reference. Raise to ignore
    # shadows/lighting, lower if real moves (esp. low-contrast ones) are missed.
    # Measured static-board noise floor is ~2; 18 was high enough that a light
    # piece leaving a light square (e.g. a pawn's source square on h2->h3) didn't
    # register, so the 2-square move showed only its destination and matched no
    # legal move. 8 stays ~4x above the noise floor.
    diff_change_threshold: float = 8.0

    # Central fraction of each square sampled for change (avoids grid lines and
    # tall neighbouring pieces leaning over the boundary).
    diff_inner_fraction: float = 0.7

    # Frames with more changed squares than this are treated as transient noise
    # (a hand passing over the board) and skipped. Castling alters 4 squares, so
    # keep this >= 4 plus a little slack.
    diff_max_changed: int = 6

    # When matching the changed-square set to a legal move, how many *extra*
    # changed squares (spurious noise) to tolerate beyond the move's own
    # squares. Every square the move alters must still be observed as changed.
    diff_tolerance: int = 1


settings = Settings()
