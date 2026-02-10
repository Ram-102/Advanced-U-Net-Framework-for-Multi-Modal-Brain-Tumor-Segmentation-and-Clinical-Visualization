import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from .data import make_generators
from .model import load_pretrained_model
from .config import TRAIN_LOG_PATH


def plot_training_curves(csv_path: str):
    hist = pd.read_csv(csv_path, sep=",", engine="python")
    epoch = range(len(hist["accuracy"]))

    f, ax = plt.subplots(1, 4, figsize=(16, 8))

    ax[0].plot(epoch, hist["accuracy"], "b", label="Training Accuracy")
    ax[0].plot(epoch, hist["val_accuracy"], "r", label="Validation Accuracy")
    ax[0].legend()

    ax[1].plot(epoch, hist["loss"], "b", label="Training Loss")
    ax[1].plot(epoch, hist["val_loss"], "r", label="Validation Loss")
    ax[1].legend()

    if "dice_coef" in hist and "val_dice_coef" in hist:
        ax[2].plot(epoch, hist["dice_coef"], "b", label="Training dice coef")
        ax[2].plot(epoch, hist["val_dice_coef"], "r", label="Validation dice coef")
        ax[2].legend()

    if "mean_io_u" in hist and "val_mean_io_u" in hist:
        ax[3].plot(epoch, hist["mean_io_u"], "b", label="Training mean IOU")
        ax[3].plot(epoch, hist["val_mean_io_u"], "r", label="Validation mean IOU")
        ax[3].legend()

    plt.show()


def main():
    _, _, test_generator, _, _, _ = make_generators()
    model = load_pretrained_model()

    results = model.evaluate(test_generator, verbose=1)
    descriptions = [
        "Loss",
        "Accuracy",
        "MeanIOU",
        "Dice coefficient",
        "Precision",
        "Sensitivity",
        "Specificity",
        "Dice coef Necrotic",
        "Dice coef Edema",
        "Dice coef Enhancing",
    ]

    print("\nModel evaluation on the test set:")
    print("==================================")
    for metric, description in zip(results, descriptions):
        print(f"{description} : {round(float(metric), 4)}")

    # Plot curves if available
    try:
        plot_training_curves(TRAIN_LOG_PATH)
    except Exception as e:
        print(f"Could not plot training curves: {e}")


if __name__ == "__main__":
    main()


