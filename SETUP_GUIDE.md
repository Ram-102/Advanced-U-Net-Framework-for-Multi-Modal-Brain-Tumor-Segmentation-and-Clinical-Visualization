# 🧠 Brain Tumor Segmentation - Setup & Usage Guide

## Quick Start (Easiest Way)

```bash
cd /Users/vechhamshivaramsrujan/Downloads/BRAIN\ TUMOR\ BY\ CNN
chmod +x start.sh
./start.sh
```

Then open your browser: **http://127.0.0.1:5000**

---

## Manual Setup (If needed)

### 1. Activate Virtual Environment
```bash
cd /Users/vechhamshivaramsrujan/Downloads/BRAIN\ TUMOR\ BY\ CNN
source venv_310/bin/activate
```

### 2. Start Flask App
```bash
python web_app/app.py
```

### 3. Access Web App
Open browser: **http://127.0.0.1:5000**

---

## Project Structure

```
BRAIN TUMOR BY CNN/
├── brats/                    # ML model code
│   ├── model.py             # U-Net model definition
│   ├── train.py             # Training script
│   ├── config.py            # Configuration
│   └── evaluate.py          # Model evaluation
├── web_app/                 # Flask web application
│   ├── app.py              # Main Flask app
│   ├── web_segmentation_direct.py  # Segmentation logic
│   ├── static/             # CSS & JavaScript
│   └── templates/          # HTML templates
├── data/                    # BraTS dataset
├── models/                  # Pre-trained model (my_model.keras)
├── outputs/                 # Generated segmentation images
├── venv_310/               # Python 3.10 virtual environment
└── start.sh                # Quick start script
```

---

## Key Files Modified/Created

### Python Files Fixed:
- ✅ `web_app/app.py` - Flask backend with proper imports
- ✅ `web_app/web_segmentation_direct.py` - Segmentation with hardmask support
- ✅ `brats/evaluate.py` - Model evaluation (runs as module)

### Frontend Files Created:
- ✅ `web_app/static/styles.css` - Added modal styling
- ✅ `web_app/templates/index.html` - Added image modal
- ✅ `web_app/static/script.js` - Fixed undefined handling

### Configuration Files:
- ✅ `.gitignore` - Version control setup
- ✅ `start.sh` - Quick start automation

---

## Running Segmentation

### 1. Select a Case
- Choose from available training/validation cases in the UI

### 2. Choose Visualization Style
- **6panel** - 6-panel comprehensive view
- **3panel** - 3-panel view (FLAIR, T1ce, prediction)
- **hardmask** - Discrete hard segmentation mask
- **comparison** - Ground truth vs prediction

### 3. Select Slices
- Enter comma-separated slice indices (e.g., "50,60,70,80")

### 4. View Results
- Click "View" button to see full-size segmentation image in modal

---

## Module Usage

### Run Evaluation
```bash
source venv_310/bin/activate
python -m brats.evaluate
```

Output:
```
Model evaluation on the test set:
==================================
Loss : 0.1034
Accuracy : 0.9778
MeanIOU : 0.7217
```

---

## Environment Details

- **Python Version**: 3.10
- **Framework**: TensorFlow 2.x
- **Web Server**: Flask
- **Architecture**: U-Net CNN
- **Dataset**: BraTS 2020
- **GPU Support**: Metal (M4 Mac optimized)

---

## Stopping the Server

```bash
# Kill Flask process
pkill -f "python web_app/app.py"
```

Or press `Ctrl+C` if running in foreground.

---

## Troubleshooting

### Port 5000 Already in Use
```bash
lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Virtual Environment Issues
```bash
# Recreate venv
rm -rf venv_310
/usr/local/bin/python3.10 -m venv venv_310
source venv_310/bin/activate
pip install -r requirements.txt
```

### Check Logs
```bash
tail -f web_app.log
```

---

## API Endpoints

- `GET /` - Main web interface
- `GET /api/health` - Health check
- `GET /api/data` - Available cases
- `POST /api/segmentation` - Run segmentation
- `GET /api/image/<filename>` - Serve segmentation image

---

## Performance Metrics

- **Model Accuracy**: 97.78%
- **Dice Coefficient**: 0.2744
- **Mean IoU**: 0.7217
- **Precision**: 0.9778
- **Sensitivity**: 0.9778
- **Specificity**: 0.9926

---

## Future Enhancements

- [ ] Add batch processing
- [ ] Export results as PDF reports
- [ ] Add more visualization options
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Add uncertainty visualization
- [ ] Real-time processing updates

---

## Support

For issues or questions, check:
1. `web_app.log` - Application logs
2. Browser console (F12) - Frontend errors
3. Terminal output - Backend errors

**Last Updated**: February 11, 2026
**Status**: ✅ Fully Working
