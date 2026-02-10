#!/usr/bin/env python3
"""Quick start script for Brain Tumor Segmentation Web App - Works anywhere after clone"""

import os
import subprocess
import sys
import time
import signal
import shutil

# Dynamically detect project directory (where this script is located)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, "venv_310")

# Find python3.10 or use system python3
PYTHON_310 = shutil.which("python3.10") or shutil.which("python3") or sys.executable

def run_command(cmd, check=True):
    """Run a shell command"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def main():
    print("🧠 Brain Tumor Segmentation Web App")
    print("=" * 50)
    print()
    
    # Check virtual environment
    if not os.path.exists(VENV_DIR):
        print("Creating Python 3.10 virtual environment...")
        success, _, err = run_command(f"{PYTHON_310} -m venv {VENV_DIR}")
        if success:
            print("✅ Virtual environment created")
        else:
            print(f"❌ Failed to create venv: {err}")
            sys.exit(1)
    else:
        print("✅ Virtual environment found")
    
    # Activate venv and check dependencies
    activate_cmd = f"source {VENV_DIR}/bin/activate"
    check_deps = f"{activate_cmd} && python -c 'import tensorflow; import flask; print(\"OK\")'"
    
    success, output, _ = run_command(check_deps, check=False)
    if output.strip() != "OK":
        print("Installing dependencies...")
        install_cmd = f"{activate_cmd} && pip install -q tensorflow flask flask-cors pandas numpy matplotlib opencv-python nibabel tqdm scikit-learn Pillow"
        success, _, err = run_command(install_cmd)
        if success:
            print("✅ Dependencies installed")
        else:
            print(f"⚠️ Installation completed with warnings")
    else:
        print("✅ All dependencies are installed")
    
    # Kill old processes
    print("\nCleaning up old processes...")
    run_command("pkill -f 'python web_app/app.py'", check=False)
    time.sleep(2)
    
    # Start Flask
    print("🚀 Starting Flask Web App...")
    os.chdir(PROJECT_DIR)
    start_cmd = f"{activate_cmd} && python web_app/app.py > web_app.log 2>&1 &"
    success, _, _ = run_command(start_cmd)
    
    time.sleep(3)
    
    # Check if running
    health_check = "curl -s http://127.0.0.1:5000/api/health"
    success, output, _ = run_command(health_check, check=False)
    
    if success and "healthy" in output:
        print("✅ Flask is running!")
        print()
        print("╔════════════════════════════════════════╗")
        print("║  🌐 Web App: http://127.0.0.1:5000    ║")
        print("║  📊 Model: Loaded and ready           ║")
        print("║  ✅ Status: Ready for segmentation    ║")
        print("╚════════════════════════════════════════╝")
        print()
        print("💡 Tip: Check web_app.log for details")
        print("Press Ctrl+C to stop the server")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⛔ Shutting down...")
            run_command("pkill -f 'python web_app/app.py'", check=False)
            print("✅ Server stopped")
    else:
        print("❌ Failed to start Flask")
        print("Check web_app.log for errors:")
        run_command("tail -20 web_app.log")
        sys.exit(1)

if __name__ == "__main__":
    main()
