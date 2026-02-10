#!/usr/bin/env python3
"""
Test script to verify web segmentation works
"""
import os
import sys
import subprocess

def test_segmentation():
    print("🧠 Testing Web Segmentation")
    print("=" * 40)
    
    # Create outputs directory
    os.makedirs('outputs/batch', exist_ok=True)
    
    # Test case
    case_id = "BraTS20_Training_017"
    style = "6panel"
    slice_idx = "50"
    
    print(f"Testing: {case_id} - {style} - slice {slice_idx}")
    
    # Run the web segmentation script
    cmd = [
        'python', 'web_segmentation.py',
        '--case', case_id,
        '--style', style,
        '--slices', slice_idx
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        # Check if image was created
        image_path = f'outputs/batch/{case_id}_s{slice_idx}_{style}.png'
        if os.path.exists(image_path):
            print(f"✅ Image created: {image_path}")
            return True
        else:
            print(f"❌ Image not found: {image_path}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_segmentation()
    if success:
        print("\n🎉 Test passed! Web segmentation is working.")
    else:
        print("\n❌ Test failed! Check the errors above.")
