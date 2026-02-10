#!/usr/bin/env python3
"""
Interactive case picker for BraTS20:
- Detects local training/validation directories
- Lists available case folders (e.g., BraTS20_Training_XXX)
- Lets you pick a case and a slice index
- Runs pretrained model (FLAIR + T1ce) and visualizes predictions
"""
import os
import sys
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brats.model import load_pretrained_model
from brats.config import IMG_SIZE, VOLUME_SLICES, VOLUME_START_AT, TRAIN_DATASET_PATH, VAL_DATASET_PATH


def find_roots() -> dict:
    return {"train": TRAIN_DATASET_PATH, "val": VAL_DATASET_PATH}


def list_cases(root: str, prefix: str) -> list:
    entries = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith(prefix):
            entries.append(name)
    return entries


def load_vol(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()


def prepare_volume_pair(case_dir: str, case_id: str) -> tuple:
    flair_path = os.path.join(case_dir, f"{case_id}_flair.nii")
    t1ce_path = os.path.join(case_dir, f"{case_id}_t1ce.nii")
    seg_path = os.path.join(case_dir, f"{case_id}_seg.nii")

    flair = load_vol(flair_path)
    t1ce = load_vol(t1ce_path)
    seg = None
    if os.path.exists(seg_path):
        seg = load_vol(seg_path)
    return flair, t1ce, seg


def build_model_input(flair: np.ndarray, t1ce: np.ndarray) -> np.ndarray:
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)
    maxv = 1e-6
    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(t1ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        maxv = max(maxv, X[j].max())
    X = X / maxv
    return X


def visualize(flair: np.ndarray, seg: np.ndarray, preds: np.ndarray, slice_idx: int):
    # preds: (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 4)
    core = preds[:, :, :, 1]
    edema = preds[:, :, :, 2]
    enhancing = preds[:, :, :, 3]

    fig, ax = plt.subplots(1, 6, figsize=(18, 5))
    base = cv2.resize(flair[:, :, slice_idx + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    for i in range(6):
        ax[i].imshow(base, cmap="gray")
        ax[i].axis("off")
    ax[0].set_title("FLAIR slice")
    if seg is not None:
        gt = cv2.resize(seg[:, :, slice_idx + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        ax[1].imshow(gt, cmap="Reds", alpha=0.3)
        ax[1].set_title("Ground truth")
    else:
        ax[1].set_title("No GT available")
    ax[2].imshow(preds[slice_idx, :, :, 1:4], cmap="Reds", alpha=0.3)
    ax[2].set_title("All classes")
    ax[3].imshow(edema[slice_idx], cmap="OrRd", alpha=0.3)
    ax[3].set_title("Edema")
    ax[4].imshow(core[slice_idx], cmap="OrRd", alpha=0.3)
    ax[4].set_title("Necrotic/Core")
    ax[5].imshow(enhancing[slice_idx], cmap="OrRd", alpha=0.3)
    ax[5].set_title("Enhancing")
    plt.tight_layout()
    plt.show()


def main():
    print("🧠 BraTS20 - Choose a case for segmentation")
    roots = find_roots()
    train_root = roots["train"]
    val_root = roots["val"]
    if not train_root and not val_root:
        print("Could not find local MICCAI_BraTS2020_TrainingData or ValidationData directories.")
        return

    train_cases = list_cases(train_root, "BraTS20_Training_") if train_root else []
    val_cases = list_cases(val_root, "BraTS20_Validation_") if val_root else []

    combined = []  # (kind, root, case_id)
    for c in train_cases:
        combined.append(("train", train_root, c))
    for c in val_cases:
        combined.append(("val", val_root, c))

    if not combined:
        print("No cases found.")
        return

    print(f"Found {len(train_cases)} training and {len(val_cases)} validation cases.")
    print("Listing up to first 50 cases:")
    for idx, (kind, _, cid) in enumerate(combined[:50]):
        print(f"  [{idx}] {kind.upper()}: {cid}")

    sel_raw = input("Enter index to segment (e.g., 0): ").strip()
    try:
        sel = int(sel_raw)
    except ValueError:
        print("Invalid index.")
        return
    if sel < 0 or sel >= len(combined):
        print("Index out of range.")
        return

    kind, root, case_id = combined[sel]
    case_dir = os.path.join(root, case_id)
    print(f"Selected: {kind.upper()} -> {case_id}")

    try:
        slice_input = input("Slice index to visualize (default 60): ").strip()
        slice_idx = int(slice_input) if slice_input else 60
    except ValueError:
        slice_idx = 60

    print("Loading volumes (FLAIR, T1ce)...")
    flair, t1ce, seg = prepare_volume_pair(case_dir, case_id)

    print("Preparing model input...")
    X = build_model_input(flair, t1ce)

    print("Loading model and predicting...")
    model = load_pretrained_model()
    preds = model.predict(X, verbose=1)

    print("Visualizing results...")
    visualize(flair, seg, preds, slice_idx)


if __name__ == "__main__":
    main()


