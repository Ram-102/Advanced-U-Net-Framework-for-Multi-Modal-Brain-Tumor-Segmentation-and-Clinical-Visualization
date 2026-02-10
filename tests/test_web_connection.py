#!/usr/bin/env python3
"""
Test the connection between web app and show_predicted_segmentations.py
"""
import os
import sys
import subprocess
import json

def test_web_connection():
    print("🧠 Testing Web App Connection")
    print("=" * 40)
    
    # Test 1: Check if show_predicted_segmentations.py works
    print("1. Testing show_predicted_segmentations.py...")
    cmd = ['python', 'show_predicted_segmentations.py', '--style', '6panel', '--slices', '50']
    
    try:
        # Simulate user input (select first case, slice 50)
        input_data = "0\n50\n"
        result = subprocess.run(cmd, input=input_data, capture_output=True, text=True, timeout=60)
        
        print(f"   Return code: {result.returncode}")
        print(f"   STDOUT: {result.stdout[:200]}...")
        if result.stderr:
            print(f"   STDERR: {result.stderr[:200]}...")
        
        if result.returncode == 0:
            print("   ✅ Script runs successfully")
        else:
            print("   ❌ Script failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Check if images are generated
    print("\n2. Checking for generated images...")
    outputs_dir = "outputs/batch"
    if os.path.exists(outputs_dir):
        images = [f for f in os.listdir(outputs_dir) if f.endswith('.png')]
        print(f"   Found {len(images)} images in {outputs_dir}")
        for img in images[:5]:  # Show first 5
            print(f"   - {img}")
    else:
        print("   ❌ No outputs/batch directory found")
        return False
    
    # Test 3: Test Flask app endpoints
    print("\n3. Testing Flask app endpoints...")
    try:
        from app import get_available_cases, get_case_index
        
        # Test get_available_cases
        cases = get_available_cases()
        print(f"   Found {len(cases)} available cases")
        
        if cases:
            # Test get_case_index
            first_case = cases[0]
            case_index = get_case_index(first_case['id'])
            print(f"   Case index for {first_case['id']}: {case_index}")
            
            if case_index is not None:
                print("   ✅ Case indexing works")
            else:
                print("   ❌ Case indexing failed")
                return False
        else:
            print("   ❌ No cases found")
            return False
            
    except Exception as e:
        print(f"   ❌ Flask app test failed: {e}")
        return False
    
    print("\n✅ All tests passed! Web app connection is working.")
    return True

if __name__ == "__main__":
    success = test_web_connection()
    if not success:
        print("\n❌ Connection test failed!")
        sys.exit(1)
