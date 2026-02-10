#!/usr/bin/env python3
"""
Startup script for the Brain Tumor Segmentation Web App
"""
import os
import sys
import subprocess
import webbrowser
import time
import threading

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_cors
        print("✅ Flask dependencies found")
        return True
    except ImportError:
        print("❌ Flask dependencies not found")
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors'])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def start_backend():
    """Start the Flask backend"""
    try:
        subprocess.run([sys.executable, 'web_app/app.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting backend: {e}")

def open_browser():
    """Open the web browser after a delay"""
    time.sleep(3)
    webbrowser.open('http://127.0.0.1:5000')

def main():
    """Main function"""
    print("🧠 Brain Tumor Segmentation Web App")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Cannot start application due to missing dependencies")
        return
    
    # Check if model exists
    if not os.path.exists('models/my_model.keras'):
        print("❌ Pre-trained model not found!")
        print("Please ensure 'models/my_model.keras' is in the current directory")
        return
    
    # Check if data exists
    if not os.path.exists('data/MICCAI_BraTS2020_TrainingData'):
        print("❌ Training data not found!")
        print("Please ensure 'data/MICCAI_BraTS2020_TrainingData' directory exists")
        return
    
    print("✅ All requirements met")
    print("🚀 Starting web application...")
    print("=" * 50)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Open browser after delay
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("🌐 Web application is starting...")
    print("📱 The application will open in your default browser")
    print("🔗 Or manually navigate to: http://127.0.0.1:5000")
    print("=" * 50)
    print("Press Ctrl+C to stop the application")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down application...")
        print("✅ Application stopped")

if __name__ == '__main__':
    main()
