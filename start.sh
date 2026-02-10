#!/bin/bash

# Brain Tumor Segmentation Web App - Quick Start Script
# Usage: ./start.sh
# Works on any machine after cloning

set -e

# Dynamically get the project directory (where this script is located)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv_310"

echo "🧠 Brain Tumor Segmentation Web App"
echo "===================================="
echo ""
echo "Project path: $PROJECT_DIR"
echo ""

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.10 venv..."
    # Try to find python3.10, fall back to python3
    PYTHON_CMD=$(command -v python3.10 || command -v python3 || echo "python3")
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "✅ Venv created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import tensorflow; import flask; print('✅ All dependencies installed')" 2>/dev/null || {
    echo "⚠️ Installing missing dependencies..."
    pip install -q pandas numpy matplotlib opencv-python nibabel tqdm flask flask-cors scikit-learn Pillow tensorflow
    echo "✅ Dependencies installed"
}

# Kill old processes
echo ""
echo "Cleaning up old processes..."
pkill -f "python web_app/app.py" 2>/dev/null || true
sleep 2

# Start Flask app
echo ""
echo "🚀 Starting Flask Web App..."
cd "$PROJECT_DIR"
python web_app/app.py > web_app.log 2>&1 &
sleep 3

# Check if running
if curl -s http://127.0.0.1:5000/api/health > /dev/null 2>&1; then
    echo "✅ Flask is running!"
    echo ""
    echo "🌐 Web App: http://127.0.0.1:5000"
    echo "📊 Model: Loaded and ready"
    echo ""
    echo "💡 Tip: Check web_app.log for details"
else
    echo "❌ Failed to start Flask"
    echo "Check web_app.log for errors"
    exit 1
fi
