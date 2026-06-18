from ultralytics import YOLO

from chessvision.settings import settings


def main():
    # Start from the COCO-pretrained nano checkpoint and fine-tune it -
    # much faster to converge than training from scratch.
    model = YOLO(str(settings.train_model_path))

    model.train(
        data=str(settings.train_data_yaml),
        epochs=settings.train_epochs,
        imgsz=settings.train_imgsz,
        patience=settings.train_patience,
        device=settings.train_device,
    )

    # Quick sanity check on the validation split.
    metrics = model.val()
    print(metrics)

    # Export the best weights to ONNX for fast inference on the Raspberry Pi.
    # Swap format="ncnn" instead if you want even faster ARM CPU inference.
    model.export(format="onnx")


# Guard the training entrypoint. On Python 3.14 the DataLoader workers are
# started via forkserver/spawn, which re-imports this module in each worker;
# keeping all work inside main() (only called here) means that re-import has no
# side effects and can't relaunch training.
if __name__ == "__main__":
    main()
