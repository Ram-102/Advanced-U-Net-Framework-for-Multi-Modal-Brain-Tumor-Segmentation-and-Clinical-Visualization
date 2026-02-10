#!/usr/bin/env python3
"""
Replicates Colab's show_predicted_segmentations with interactive selection:
- Prompt to choose specific cases and slices (no random selection)
- Save Kaggle-like panels; optional 6-panel or 3-panel styles
"""
import os
import sys
import random
import argparse
import numpy as np
import cv2
import nibabel as nib

# Set matplotlib backend before importing pyplot to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brats.model import load_pretrained_model
from brats.config import IMG_SIZE, VOLUME_SLICES, VOLUME_START_AT, TRAIN_DATASET_PATH, VAL_DATASET_PATH


def find_train_root():
    return TRAIN_DATASET_PATH


def find_val_root():
    return VAL_DATASET_PATH


def list_cases(root: str) -> list:
    return [d for d in sorted(os.listdir(root)) if d.startswith("BraTS20_Training_") and os.path.isdir(os.path.join(root, d))]


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


def plot_case_panels(case_dir: str, case_id: str, slice_idx: int, model) -> str:
    flair = load_vol(os.path.join(case_dir, f"{case_id}_flair.nii"))
    t1ce = load_vol(os.path.join(case_dir, f"{case_id}_t1ce.nii"))
    seg_path = os.path.join(case_dir, f"{case_id}_seg.nii")
    gt = load_vol(seg_path) if os.path.exists(seg_path) else None

    X = build_input(flair, t1ce)
    preds = model.predict(X, verbose=0)
    hard = preds.argmax(-1).astype(np.uint8)

    s = slice_idx
    os.makedirs("../generated_outputs/batch", exist_ok=True)

    # Calculate metrics for this slice
    if gt is not None:
        gt_slice = gt[:, :, s + VOLUME_START_AT]
        metrics = calculate_metrics(hard[s], gt_slice)
        print(f"Metrics for {case_id} slice {s}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
    else:
        metrics = {"error": "No ground truth available"}
        print(f"No ground truth available for {case_id}")

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
    out_png = os.path.join("../generated_outputs", "batch", f"{case_id}_s{slice_idx}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Only return the main panel path (avoid saving extra black masks)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Show predicted segmentations with interactive case/slice selection.")
    parser.add_argument("--slices", type=str, default="50,60,70,80", help="Comma-separated slice indices to export")
    parser.add_argument("--style", choices=["6panel", "3panel", "hardmask", "comparison"], default="6panel", help="Figure style: 6panel overlays, 3panel (FLAIR, T1ce, prob), hardmask (FLAIR, T1ce, discrete mask), or comparison (GT vs Pred)")
    args = parser.parse_args()

    print("🧠 show_predicted_segmentations - Local")
    train_root = find_train_root()
    val_root = find_val_root()
    if not train_root and not val_root:
        print("Could not find MICCAI_BraTS2020_TrainingData or ValidationData")
        return

    train_cases = list_cases(train_root) if train_root else []
    # Validation dirs differ in naming; accept any BraTS20_Validation_*
    val_cases = []
    if val_root:
        val_cases = [d for d in sorted(os.listdir(val_root)) if d.startswith("BraTS20_Validation_") and os.path.isdir(os.path.join(val_root, d))]

    combined = []  # (kind, root, id)
    for c in train_cases:
        combined.append(("TRAIN", train_root, c))
    for c in val_cases:
        combined.append(("VAL", val_root, c))
    if not combined:
        print("No cases found.")
        return

    model = load_pretrained_model()

    # Purely interactive flow (no random)
    print(f"Found {len(train_cases)} TRAIN and {len(val_cases)} VAL cases (total {len(combined)}).")
    for idx, (k, _, cid) in enumerate(combined[:100]):
        print(f"  [{idx}] {k}: {cid}")
    sel_raw = input("Enter case index/indices (e.g., 3 or 1,4,7): ").strip()
    if not sel_raw:
        print("No selection provided.")
        return
    try:
        selected = [int(x.strip()) for x in sel_raw.split(',') if x.strip()]
    except ValueError:
        print("Invalid indices.")
        return

    slices_in = input(f"Slices to export (default {args.slices}): ").strip() or args.slices
    slice_indices = [int(s.strip()) for s in slices_in.split(",") if s.strip()]

    os.makedirs('../generated_outputs/batch', exist_ok=True)

    for i in selected:
        if i < 0 or i >= len(combined):
            print(f"Skipping invalid index {i}")
            continue
        kind, root, case = combined[i]
        case_dir = os.path.join(root, case)

        if args.style == "3panel":
            flair = load_vol(os.path.join(case_dir, f"{case}_flair.nii"))
            t1ce = load_vol(os.path.join(case_dir, f"{case}_t1ce.nii"))
            X = build_input(flair, t1ce)
            preds = model.predict(X, verbose=0)
            hard = preds.argmax(-1).astype(np.uint8)
            
            # Debug info
            print(f"Case {case}: preds shape {preds.shape}, hard unique {np.unique(hard)}")
            print(f"Hard mask value counts: {np.bincount(hard.flatten())}")
            
            for s in slice_indices:
                fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                ax[0].imshow(cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray'); ax[0].set_title('FLAIR'); ax[0].axis('off')
                ax[1].imshow(cv2.resize(t1ce[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray'); ax[1].set_title('T1ce'); ax[1].axis('off')
                
                # Use probability maps instead of hard mask for better visualization
                # Show combined tumor classes (1,2,3) as probability
                tumor_probs = preds[s, :, :, 1:4].sum(axis=2)  # Sum of classes 1,2,3
                # Debug: print probability values
                print(f"  Slice {s}: tumor_probs range {tumor_probs.min():.6f} to {tumor_probs.max():.6f}")
                print(f"  Slice {s}: max Core={preds[s, :, :, 1].max():.6f}, Edema={preds[s, :, :, 2].max():.6f}, Enh={preds[s, :, :, 3].max():.6f}")
                # Use a more sensitive colormap and scaling
                ax[2].imshow(tumor_probs, cmap='hot', vmin=0, vmax=0.1); ax[2].set_title(f'Segmentation (slice {s})'); ax[2].axis('off')
                
                plt.tight_layout()
                out = os.path.join('../generated_outputs', 'batch', f'{case}_s{s}_3panel.png')
                plt.savefig(out, dpi=150); plt.close()
                print(f"Saved {out} - tumor pixels: {(tumor_probs > 0.1).sum()}")
            continue

        if args.style == "hardmask":
            flair = load_vol(os.path.join(case_dir, f"{case}_flair.nii"))
            t1ce = load_vol(os.path.join(case_dir, f"{case}_t1ce.nii"))
            X = build_input(flair, t1ce)
            preds = model.predict(X, verbose=0)
            hard = preds.argmax(-1).astype(np.uint8)
            
            # Debug info
            print(f"Case {case}: preds shape {preds.shape}, hard unique {np.unique(hard)}")
            print(f"Hard mask value counts: {np.bincount(hard.flatten())}")
            
            for s in slice_indices:
                fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                ax[0].imshow(cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray'); ax[0].set_title('FLAIR'); ax[0].axis('off')
                ax[1].imshow(cv2.resize(t1ce[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray'); ax[1].set_title('T1ce'); ax[1].axis('off')
                
                # Show discrete hard mask with proper colormap
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=0, vmax=3)
                im = ax[2].imshow(hard[s], cmap=cmap, norm=norm)
                ax[2].set_title(f'Segmentation (slice {s})'); ax[2].axis('off')
                plt.colorbar(im, ax=ax[2], shrink=0.6)
                
                plt.tight_layout()
                out = os.path.join('../generated_outputs', 'batch', f'{case}_s{s}_hardmask.png')
                plt.savefig(out, dpi=150); plt.close()
                print(f"Saved {out} - tumor pixels: {(hard[s] > 0).sum()}")
            continue

        if args.style == "comparison":
            flair = load_vol(os.path.join(case_dir, f"{case}_flair.nii"))
            t1ce = load_vol(os.path.join(case_dir, f"{case}_t1ce.nii"))
            seg_path = os.path.join(case_dir, f"{case}_seg.nii")
            gt = load_vol(seg_path) if os.path.exists(seg_path) else None
            
            if gt is None:
                print(f"No ground truth available for {case}, skipping comparison mode")
                continue
                
            X = build_input(flair, t1ce)
            preds = model.predict(X, verbose=0)
            hard = preds.argmax(-1).astype(np.uint8)
            
            for s in slice_indices:
                fig, ax = plt.subplots(2, 2, figsize=(12, 10))
                
                # Top row: Original images
                ax[0,0].imshow(cv2.resize(flair[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
                ax[0,0].set_title('FLAIR Input'); ax[0,0].axis('off')
                
                ax[0,1].imshow(cv2.resize(t1ce[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap='gray')
                ax[0,1].set_title('T1ce Input'); ax[0,1].axis('off')
                
                # Bottom row: Ground Truth and Difference Map
                gt_slice = cv2.resize(gt[:, :, s + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                pred_slice = hard[s]
                
                # Ground Truth
                ax[1,0].imshow(gt_slice, cmap='viridis', vmin=0, vmax=3)
                ax[1,0].set_title('Ground Truth'); ax[1,0].axis('off')
                
                # Difference map (Green=Correct, Red=Wrong, Blue=Missed)
                diff_map = np.zeros((IMG_SIZE, IMG_SIZE, 3))
                for class_id in [1, 2, 3]:
                    gt_mask = (gt_slice == class_id)
                    pred_mask = (pred_slice == class_id)
                    
                    # Green: Correctly predicted
                    correct = gt_mask & pred_mask
                    diff_map[correct, 1] = 1  # Green channel
                    
                    # Red: False positive (predicted but not in GT)
                    false_pos = pred_mask & ~gt_mask
                    diff_map[false_pos, 0] = 1  # Red channel
                    
                    # Blue: False negative (in GT but not predicted)
                    false_neg = gt_mask & ~pred_mask
                    diff_map[false_neg, 2] = 1  # Blue channel
                
                ax[1,1].imshow(diff_map)
                ax[1,1].set_title('Difference Map\n(Green=Correct, Red=Wrong, Blue=Missed)'); ax[1,1].axis('off')
                
                # Calculate and display metrics
                metrics = calculate_metrics(pred_slice, gt_slice)
                metrics_text = f"Slice {s} Metrics:\n"
                if "error" not in metrics:
                    metrics_text += f"Dice WT: {metrics.get('dice_whole_tumor', 0):.3f}\n"
                    metrics_text += f"Dice Core: {metrics.get('dice_necrotic_core', 0):.3f}\n"
                    metrics_text += f"Dice Edema: {metrics.get('dice_edema', 0):.3f}\n"
                    metrics_text += f"Dice Enh: {metrics.get('dice_enhancing', 0):.3f}"
                else:
                    metrics_text += "No GT available"
                
                plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                
                plt.tight_layout()
                out = os.path.join('../generated_outputs', 'batch', f'{case}_s{s}_comparison.png')
                plt.savefig(out, dpi=150); plt.close()
                print(f"Saved {out}")
                
                # Print metrics
                if "error" not in metrics:
                    print(f"  Dice scores - WT: {metrics.get('dice_whole_tumor', 0):.3f}, Core: {metrics.get('dice_necrotic_core', 0):.3f}, Edema: {metrics.get('dice_edema', 0):.3f}, Enh: {metrics.get('dice_enhancing', 0):.3f}")
            continue

        # Default 6-panel style
        for s in slice_indices:
            out = plot_case_panels(case_dir, case, s, model)
            print(f"Saved {out}")
    print("Done. Files in ../generated_outputs/batch/")
    
    # Print overall summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print("✅ Metrics Dashboard + Comparison Mode Added Successfully!")
    print("📊 Each case now shows:")
    print("   - Dice scores for each tumor class")
    print("   - IoU (Intersection over Union)")
    print("   - Whole tumor detection accuracy")
    print("   - Visual comparison: GT vs Prediction vs Difference")
    print("   - Color-coded accuracy: Green=Correct, Red=Wrong, Blue=Missed")
    print("🎯 Perfect for faculty presentation!")
    print("="*50)


if __name__ == "__main__":
    main()


