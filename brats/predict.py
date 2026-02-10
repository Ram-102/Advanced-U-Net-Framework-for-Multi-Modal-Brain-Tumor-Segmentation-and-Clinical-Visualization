import os
import random
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt

from .config import (
    TRAIN_DATASET_PATH,
    IMG_SIZE,
    VOLUME_SLICES,
    VOLUME_START_AT,
    SEGMENT_CLASSES,
)
from .model import load_pretrained_model


def predict_by_path(model, case_path, case_id):
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)

    flair = nib.load(os.path.join(case_path, f"{case_id}_flair.nii")).get_fdata()
    ce = nib.load(os.path.join(case_path, f"{case_id}_t1ce.nii")).get_fdata()

    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

    return model.predict(X / max(np.max(X), 1e-6), verbose=0)


def show_predicts_by_id(model, case_id, start_slice=60):
    path = os.path.join(TRAIN_DATASET_PATH, f"{case_id}")
    gt = nib.load(os.path.join(path, f"{case_id}_seg.nii")).get_fdata()
    orig = nib.load(os.path.join(path, f"{case_id}_flair.nii")).get_fdata()
    p = predict_by_path(model, path, case_id)

    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]

    f, axarr = plt.subplots(1, 6, figsize=(18, 6))
    for i in range(6):
        axarr[i].imshow(cv2.resize(orig[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")

    axarr[0].imshow(cv2.resize(orig[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].set_title("Original image flair")
    curr_gt = cv2.resize(gt[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation="none", alpha=0.3)
    axarr[1].set_title("Ground truth")
    axarr[2].imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation="none", alpha=0.3)
    axarr[2].set_title("All classes predicted")
    axarr[3].imshow(edema[start_slice, :, :], cmap="OrRd", interpolation="none", alpha=0.3)
    axarr[3].set_title(f"{SEGMENT_CLASSES[1]} predicted")
    axarr[4].imshow(core[start_slice, :, :], cmap="OrRd", interpolation="none", alpha=0.3)
    axarr[4].set_title(f"{SEGMENT_CLASSES[2]} predicted")
    axarr[5].imshow(enhancing[start_slice, :, :], cmap="OrRd", interpolation="none", alpha=0.3)
    axarr[5].set_title(f"{SEGMENT_CLASSES[3]} predicted")
    plt.tight_layout()
    plt.show()


def main():
    model = load_pretrained_model()

    # Pick three random cases to visualize
    case_dirs = [d for d in os.listdir(TRAIN_DATASET_PATH) if d.startswith("BraTS20_Training_")]
    random.shuffle(case_dirs)
    for case_dir in case_dirs[:3]:
        print(f"Visualizing predictions for {case_dir}")
        show_predicts_by_id(model, case_dir, start_slice=60)


if __name__ == "__main__":
    main()


