#!/usr/bin/env python3
"""
Simple script to run the Brain Tumor Segmentation Web App
"""
import os
import sys
import webbrowser
import time
import subprocess

def main():
    print("🧠 Brain Tumor Segmentation Web App")
    print("=" * 50)
    
    # Get project root (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define paths
    model_path = os.path.join(project_root, 'models', 'my_model.keras')
    data_path = os.path.join(project_root, 'data', 'MICCAI_BraTS2020_TrainingData')
    app_path = os.path.join(project_root, 'web_app', 'app.py')
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Pre-trained model not found!")
        print(f"Please ensure '{model_path}' exists")
        return
    
    # Check if data exists
    if not os.path.exists(data_path):
        print("❌ Training data not found!")
        print(f"Please ensure '{data_path}' directory exists")
        return
    
    print("✅ All requirements met")
    print("🚀 Starting web application...")
    print("=" * 50)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://127.0.0.1:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the Flask app
    try:
        subprocess.run([sys.executable, app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down application...")
        print("✅ Application stopped")

if __name__ == '__main__':
    main()
