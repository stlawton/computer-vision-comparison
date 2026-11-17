
from pathlib import Path
from shutil import copy2
from ultralytics import YOLO


def main():
    # Load pretrained YOLO26 nano model.
    model = YOLO("yolo26n.pt")

    # Fine-tune on your labeled dataset.
    model.train(
        data="Datasets/data.yaml",
        epochs=50,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        cache=False,
        save=True,
        project="runs/training",
        name="yolo26n_v1",

        # Disable lighting/color augmentation.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # Disable geometric augmentation.
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
    )

    # Locate the best trained weights.
    best_model = Path(model.trainer.best)

    # Save a permanent copy.
    saved_model = Path("Models/yolo26n_finetuned.pt")
    saved_model.parent.mkdir(parents=True, exist_ok=True)

    copy2(best_model, saved_model)

    print(f"Trained model saved to: {saved_model}")


if __name__ == "__main__":
    main()
