from pathlib import Path
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
    # types x {white, black}), used for inference in detect_pieces.py.
    pieces_model_path: Path = Path("runs/detect/train/weights/best.pt")

    # Base checkpoint to fine-tune from in train_pieces.py. yolo26n.pt is the
    # COCO-pretrained nano model - much faster to converge than from scratch.
    train_model_path: Path = Path("yolo26n.pt")

    # data.yaml that came with the chess-pieces dataset (YOLO format).
    train_data_yaml: Path = Path("datasets/chess-pieces/data.yaml")

    # Training hyperparameters for train_pieces.py.
    train_epochs: int = 100
    train_imgsz: int = 640
    train_patience: int = 20  # stop early if val performance plateaus
    train_device: int = 0  # GPU index; set if you add an NVIDIA/CUDA GPU

    # If the warped board's top-left corner corresponds to square a8 from the
    # camera's point of view, leave this False. Flip to True if your camera/
    # board orientation puts a1 in the top-left instead.
    flip_orientation: bool = True

    # The warped board is only as big as the board appears in the raw camera
    # frame, which can be quite small. Scale the displayed view up to at least
    # this many pixels on a side so detected pieces/boxes are easy to see.
    display_size: int = 800


settings = Settings()
