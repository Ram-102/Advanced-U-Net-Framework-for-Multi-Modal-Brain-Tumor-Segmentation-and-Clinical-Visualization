#!/bin/bash

# Brain Tumor Segmentation Web App - Quick Start Script
# Usage: ./start.sh

set -e

PROJECT_DIR="/Users/vechhamshivaramsrujan/Downloads/BRAIN TUMOR BY CNN"
VENV_DIR="$PROJECT_DIR/venv_310"

echo "🧠 Brain Tumor Segmentation Web App"
echo "===================================="
echo ""

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating Python 3.10 venv..."
    /usr/local/bin/python3.10 -m venv "$VENV_DIR"
    echo "✅ Venv created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated (Python 3.10)"

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
