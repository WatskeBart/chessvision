from ultralytics import YOLO

# Path to the data.yaml that came with your chess-pieces dataset
# (e.g. downloaded from Roboflow Universe in YOLO format).
DATA_YAML = "datasets/chess-pieces/data.yaml"

# Start from the COCO-pretrained nano checkpoint and fine-tune it -
# much faster to converge than training from scratch.
model = YOLO("yolo26n.pt")

results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    patience=20,   # stop early if val performance plateaus
    device="cpu",  # set to 0 if you have an NVIDIA GPU available
)

# Quick sanity check on the validation split
metrics = model.val()
print(metrics)

# Export the best weights to ONNX for fast inference on the Raspberry Pi.
# Swap format="ncnn" instead if you want even faster ARM CPU inference.
model.export(format="onnx")
