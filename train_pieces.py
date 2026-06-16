from ultralytics import YOLO

from settings import settings

# Start from the COCO-pretrained nano checkpoint and fine-tune it -
# much faster to converge than training from scratch.
model = YOLO(str(settings.train_model_path))

# Guard the training entrypoint. On Python 3.14 the DataLoader workers are
# started via forkserver/spawn, which re-imports this module in each worker;
# without this guard that re-import would relaunch training recursively.
if __name__ == "__main__":
    results = model.train(
        data=str(settings.train_data_yaml),
        epochs=settings.train_epochs,
        imgsz=settings.train_imgsz,
        patience=settings.train_patience,
        device=settings.train_device,
    )

    # Quick sanity check on the validation split
    metrics = model.val()
    print(metrics)

    # Export the best weights to ONNX for fast inference on the Raspberry Pi.
    # Swap format="ncnn" instead if you want even faster ARM CPU inference.
    model.export(format="onnx")
