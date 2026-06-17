from ultralytics import YOLO

# Path to the data.yaml that came with your chess-pieces dataset
# (e.g. downloaded from Roboflow Universe in YOLO format).
DATA_YAML = "datasets/chess-pieces/data.yaml"

# Start from the COCO-pretrained nano checkpoint and fine-tune it -
# much faster to converge than training from scratch.
model = YOLO("yolo26n.pt")

# Guard the training entrypoint. On Python 3.14 the DataLoader workers are
# started via forkserver/spawn, which re-imports this module in each worker;
# without this guard that re-import would relaunch training recursively.
if __name__ == "__main__":
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        patience=20,   # stop early if val performance plateaus
        device=0  # no NVIDIA GPU on this machine (AMD iGPU, no CUDA/ROCm) - set to 0 if you add one
    )

    # Quick sanity check on the validation split
    metrics = model.val()
    print(metrics)

    # Export the best weights to ONNX for fast inference on the Raspberry Pi.
    # Swap format="ncnn" instead if you want even faster ARM CPU inference.
    model.export(format="onnx")
