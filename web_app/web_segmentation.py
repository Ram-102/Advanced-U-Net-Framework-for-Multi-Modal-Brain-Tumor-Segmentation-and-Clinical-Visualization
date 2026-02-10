#!/usr/bin/env python3
"""
Web-compatible segmentation script that runs without interactive prompts
"""
import os
import sys
import argparse
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brats.model import load_pretrained_model
from brats.config import IMG_SIZE, VOLUME_SLICES, VOLUME_START_AT, TRAIN_DATASET_PATH, VAL_DATASET_PATH


def find_train_root():
    return TRAIN_DATASET_PATH


def find_val_root():
    return VAL_DATASET_PATH


def load_vol(path: str):
    return nib.load(path).get_fdata()


def build_input(flair, t1ce):
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)
    maxv = 1e-6
    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(t1ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        maxv = max(maxv, X[j].max())
    return X / maxv


def calculate_metrics(pred, gt):
    """Calculate Dice scores and other metrics"""
    metrics = {}
    if gt is None:
        return {"error": "No ground truth available"}
    
    # Resize GT to match prediction
    gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Calculate Dice for each class (1,2,3)
    class_names = {1: 'Necrotic/Core', 2: 'Edema', 3: 'Enhancing'}
    for class_id, class_name in class_names.items():
        pred_mask = (pred == class_id)
        gt_mask = (gt_resized == class_id)
        
        intersection = (pred_mask & gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice = (2.0 * intersection) / (union + 1e-6)
        metrics[f'dice_{class_name.lower().replace("/", "_")}'] = dice
        
        # Calculate IoU
        iou = intersection / (union - intersection + 1e-6)
        metrics[f'iou_{class_name.lower().replace("/", "_")}'] = iou
    
    # Overall tumor detection
    pred_tumor = (pred > 0)
    gt_tumor = (gt_resized > 0)
    intersection = (pred_tumor & gt_tumor).sum()
    union = pred_tumor.sum() + gt_tumor.sum()
    metrics['dice_whole_tumor'] = (2.0 * intersection) / (union + 1e-6)
    
    return metrics


def run_segmentation_for_case(case_id, style, slice_indices):
    """Run segmentation for a specific case and style"""
    # Find the case directory
    train_root = find_train_root()
    val_root = find_val_root()
    
    case_dir = None
    if train_root:
        train_case_dir = os.path.join(train_root, case_id)
        if os.path.exists(train_case_dir):
            case_dir = train_case_dir
    
    if not case_dir and val_root:
        val_case_dir = os.path.join(val_root, case_id)
        if os.path.exists(val_case_dir):
            case_dir = val_case_dir
    
    if not case_dir:
        print(f"Case {case_id} not found")
        return False
    
    # Load model
    model = load_pretrained_model()
    
    # Load data
    flair = load_vol(os.path.join(case_dir, f"{case_id}_flair.nii"))
    t1ce = load_vol(os.path.join(case_dir, f"{case_id}_t1ce.nii"))
    seg_path = os.path.join(case_dir, f"{case_id}_seg.nii")
    gt = load_vol(seg_path) if os.path.exists(seg_path) else None
    
    # Build input
    X = build_input(flair, t1ce)
    preds = model.predict(X, verbose=0)
    hard = preds.argmax(-1).astype(np.uint8)
    
    # Create outputs directory
    os.makedirs("../generated_outputs/batch", exist_ok=True)
    
    # Generate images for each slice
    for slice_idx in slice_indices:
        try:
            if style == "6panel":
                generate_6panel(flair, t1ce, gt, preds, hard, case_id, slice_idx)
            elif style == "3panel":
                generate_3panel(flair, t1ce, preds, case_id, slice_idx)
            elif style == "hardmask":
                generate_hardmask(flair, t1ce, hard, case_id, slice_idx)
            elif style == "comparison":
                generate_comparison(flair, t1ce, gt, hard, case_id, slice_idx)
            
            print(f"Generated {case_id}_s{slice_idx}_{style}.png")
            
        except Exception as e:
            print(f"Error generating {case_id}_s{slice_idx}_{style}.png: {e}")
            return False
    
    return True


def generate_6panel(flair, t1ce, gt, preds, hard, case_id, slice_idx):
    """Generate 6-panel visualization"""
    s = slice_idx
    fig, ax = plt.subplots(1, 6, figsize=(18, 5))
    base = cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    for a in ax:
        a.imshow(base, cmap="gray")
        a.axis("off")
    ax[0].set_title("Background (FLAIR)")

    if gt is not None:
        gts = cv2.resize(gt[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        ax[1].imshow(gts, cmap="Reds", alpha=0.35)
        ax[1].set_title("Ground truth")
    else:
        ax[1].set_title("Ground truth (N/A)")

    ax[2].imshow(preds[s, :, :, 1:4], cmap="Reds", alpha=0.35)
    ax[2].set_title("Predicted - all classes")
    ax[3].imshow(preds[s, :, :, 2], cmap="inferno", alpha=0.7)
    ax[3].set_title("Edema")
    ax[4].imshow(preds[s, :, :, 1], cmap="magma", alpha=0.7)
    ax[4].set_title("Necrotic/Core")
    ax[5].imshow(preds[s, :, :, 3], cmap="viridis", alpha=0.7)
    ax[5].set_title("Enhancing")
    plt.tight_layout()
    
    out_png = os.path.join("../generated_outputs", "batch", f"{case_id}_s{slice_idx}_6panel.png")
    plt.savefig(out_png, dpi=150)
    plt.close()


def generate_3panel(flair, t1ce, preds, case_id, slice_idx):
    """Generate 3-panel visualization"""
    s = slice_idx
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
    ax[0].set_title('FLAIR')
    ax[0].axis('off')
    ax[1].imshow(cv2.resize(t1ce[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
    ax[1].set_title('T1ce')
    ax[1].axis('off')
    
    # Show combined tumor classes (1,2,3) as probability
    tumor_probs = preds[s, :, :, 1:4].sum(axis=2)
    ax[2].imshow(tumor_probs, cmap='hot', vmin=0, vmax=0.1)
    ax[2].set_title(f'Segmentation (slice {s})')
    ax[2].axis('off')
    
    plt.tight_layout()
    out = os.path.join('../generated_outputs', 'batch', f'{case_id}_s{s}_3panel.png')
    plt.savefig(out, dpi=150)
    plt.close()


def generate_hardmask(flair, t1ce, hard, case_id, slice_idx):
    """Generate hard mask visualization"""
    s = slice_idx
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
    ax[0].set_title('FLAIR')
    ax[0].axis('off')
    ax[1].imshow(cv2.resize(t1ce[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
    ax[1].set_title('T1ce')
    ax[1].axis('off')
    
    # Show discrete hard mask
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=3)
    im = ax[2].imshow(hard[s], cmap=cmap, norm=norm)
    ax[2].set_title(f'Segmentation (slice {s})')
    ax[2].axis('off')
    plt.colorbar(im, ax=ax[2], shrink=0.6)
    
    plt.tight_layout()
    out = os.path.join('../generated_outputs', 'batch', f'{case_id}_s{s}_hardmask.png')
    plt.savefig(out, dpi=150)
    plt.close()


def generate_comparison(flair, t1ce, gt, hard, case_id, slice_idx):
    """Generate comparison visualization"""
    s = slice_idx
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Original images
    ax[0,0].imshow(cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
    ax[0,0].set_title('FLAIR Input')
    ax[0,0].axis('off')
    
    ax[0,1].imshow(cv2.resize(t1ce[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
    ax[0,1].set_title('T1ce Input')
    ax[0,1].axis('off')
    
    # Combined input
    combined = np.stack([
        cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)),
        cv2.resize(t1ce[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    ], axis=-1)
    ax[0,2].imshow(combined[:,:,0], cmap='gray', alpha=0.7)
    ax[0,2].imshow(combined[:,:,1], cmap='hot', alpha=0.3)
    ax[0,2].set_title('Combined Input')
    ax[0,2].axis('off')
    
    # Bottom row: Ground Truth and Difference
    if gt is not None:
        gt_slice = cv2.resize(gt[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        pred_slice = hard[s]
        
        # Ground Truth
        ax[1,0].imshow(gt_slice, cmap='viridis', vmin=0, vmax=3)
        ax[1,0].set_title('Ground Truth')
        ax[1,0].axis('off')
        
        # Difference map
        diff_map = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        for class_id in [1, 2, 3]:
            gt_mask = (gt_slice == class_id)
            pred_mask = (pred_slice == class_id)
            
            # Green: Correctly predicted
            correct = gt_mask & pred_mask
            diff_map[correct, 1] = 1
            
            # Red: False positive
            false_pos = pred_mask & ~gt_mask
            diff_map[false_pos, 0] = 1
            
            # Blue: False negative
            false_neg = gt_mask & ~pred_mask
            diff_map[false_neg, 2] = 1
        
        ax[1,1].imshow(diff_map)
        ax[1,1].set_title('Difference Map\n(Green=Correct, Red=Wrong, Blue=Missed)')
        ax[1,1].axis('off')
    else:
        ax[1,0].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', transform=ax[1,0].transAxes)
        ax[1,1].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', transform=ax[1,1].transAxes)
    
    # Hide the third panel
    ax[1,2].axis('off')
    
    plt.tight_layout()
    out = os.path.join('../generated_outputs', 'batch', f'{case_id}_s{s}_comparison.png')
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Web-compatible segmentation")
    parser.add_argument("--case", required=True, help="Case ID to process")
    parser.add_argument("--style", required=True, help="Visualization style")
    parser.add_argument("--slices", required=True, help="Comma-separated slice indices")
    
    args = parser.parse_args()
    
    slice_indices = [int(s.strip()) for s in args.slices.split(',') if s.strip()]
    
    success = run_segmentation_for_case(args.case, args.style, slice_indices)
    
    if success:
        print("Segmentation completed successfully")
        sys.exit(0)
    else:
        print("Segmentation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
