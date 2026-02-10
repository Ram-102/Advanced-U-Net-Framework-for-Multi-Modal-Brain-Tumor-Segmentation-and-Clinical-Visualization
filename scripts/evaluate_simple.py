#!/usr/bin/env python3
"""
Simple evaluation script without matplotlib dependency
"""
import os
import sys
import numpy as np
import cv2
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brats.model import load_pretrained_model
from brats.config import IMG_SIZE, VOLUME_SLICES, VOLUME_START_AT, TRAIN_DATASET_PATH

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
    dice_scores = []
    
    for class_id, class_name in class_names.items():
        pred_mask = (pred == class_id)
        gt_mask = (gt_resized == class_id)
        
        intersection = (pred_mask & gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice = (2.0 * intersection) / (union + 1e-6)
        dice_scores.append(dice)
        
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
    
    # Calculate overall Dice
    metrics['dice_overall'] = np.mean(dice_scores)
    
    return metrics

def find_train_root():
    return TRAIN_DATASET_PATH

def evaluate_model():
    """Evaluate the model on available training cases"""
    print("Loading pre-trained model...")
    model = load_pretrained_model()
    print("✅ Model loaded successfully!")
    
    train_root = find_train_root()
    if not train_root:
        print("❌ Training data not found!")
        return
    
    print(f"📁 Found training data at: {train_root}")
    
    # Get list of cases
    cases = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    cases = sorted(cases)[:5]  # Test on first 5 cases
    
    print(f"🧠 Testing on {len(cases)} cases...")
    
    all_metrics = []
    
    for case in cases:
        case_dir = os.path.join(train_root, case)
        print(f"\n📊 Processing {case}...")
        
        try:
            # Load data
            flair_path = os.path.join(case_dir, f"{case}_flair.nii")
            t1ce_path = os.path.join(case_dir, f"{case}_t1ce.nii")
            seg_path = os.path.join(case_dir, f"{case}_seg.nii")
            
            if not all(os.path.exists(p) for p in [flair_path, t1ce_path, seg_path]):
                print(f"⚠️  Missing files for {case}, skipping...")
                continue
            
            flair = load_vol(flair_path)
            t1ce = load_vol(t1ce_path)
            gt = load_vol(seg_path)
            
            # Build input
            X = build_input(flair, t1ce)
            
            # Predict
            preds = model.predict(X, verbose=0)
            hard = preds.argmax(-1).astype(np.uint8)
            
            # Calculate metrics for middle slice
            slice_idx = 50  # Middle slice
            pred_slice = hard[slice_idx]
            gt_slice = gt[:, :, slice_idx + VOLUME_START_AT]
            
            metrics = calculate_metrics(pred_slice, gt_slice)
            
            if "error" not in metrics:
                all_metrics.append(metrics)
                print(f"  Dice Overall: {metrics['dice_overall']:.3f}")
                print(f"  Dice Necrotic: {metrics['dice_necrotic_core']:.3f}")
                print(f"  Dice Edema: {metrics['dice_edema']:.3f}")
                print(f"  Dice Enhancing: {metrics['dice_enhancing']:.3f}")
                print(f"  Dice Whole Tumor: {metrics['dice_whole_tumor']:.3f}")
            else:
                print(f"  ❌ Error: {metrics['error']}")
                
        except Exception as e:
            print(f"  ❌ Error processing {case}: {e}")
    
    # Calculate average metrics
    if all_metrics:
        print(f"\n📈 AVERAGE PERFORMANCE ACROSS {len(all_metrics)} CASES:")
        print("=" * 50)
        
        avg_dice_overall = np.mean([m['dice_overall'] for m in all_metrics])
        avg_dice_necrotic = np.mean([m['dice_necrotic_core'] for m in all_metrics])
        avg_dice_edema = np.mean([m['dice_edema'] for m in all_metrics])
        avg_dice_enhancing = np.mean([m['dice_enhancing'] for m in all_metrics])
        avg_dice_whole = np.mean([m['dice_whole_tumor'] for m in all_metrics])
        
        print(f"Overall Dice Coefficient: {avg_dice_overall:.3f}")
        print(f"Necrotic Core Dice:       {avg_dice_necrotic:.3f}")
        print(f"Edema Dice:              {avg_dice_edema:.3f}")
        print(f"Enhancing Tumor Dice:    {avg_dice_enhancing:.3f}")
        print(f"Whole Tumor Dice:        {avg_dice_whole:.3f}")
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"✅ Excellent (>0.8): {sum(1 for m in all_metrics if m['dice_overall'] > 0.8)} cases")
        print(f"✅ Good (0.7-0.8):   {sum(1 for m in all_metrics if 0.7 <= m['dice_overall'] <= 0.8)} cases")
        print(f"⚠️  Fair (0.6-0.7):   {sum(1 for m in all_metrics if 0.6 <= m['dice_overall'] < 0.7)} cases")
        print(f"❌ Poor (<0.6):      {sum(1 for m in all_metrics if m['dice_overall'] < 0.6)} cases")
    else:
        print("❌ No valid metrics calculated!")

if __name__ == "__main__":
    evaluate_model()
