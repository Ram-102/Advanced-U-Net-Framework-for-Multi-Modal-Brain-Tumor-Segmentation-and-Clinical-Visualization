# Brain Tumor Segmentation (BraTS20) - Modular Python Project

This project implements a U-Net model for brain tumor segmentation using the BraTS 2020 dataset. The project has been organized into modular components for better maintainability and ease of use.

## Project Structure

- **`brats/`**: Core library code
  - `config.py`: Configuration and paths
  - `model.py`: U-Net model architecture
  - `data.py`: Data loading and processing
  - `metrics.py`: Evaluation metrics
- **`web_app/`**: Flask-based web application
  - `app.py`: Backend server
  - `templates/`: HTML templates
  - `static/`: CSS and JS files
- **`scripts/`**: Utility scripts
  - `run_web_app.py`: Simple launcher for the web app
  - `evaluate_simple.py`: Basic model evaluation
  - `show_predicted_segmentations.py`: Visualization tool
- **`data/`**: Dataset directory (BraTS 2020)
- **`models/`**: Trained models (e.g., `my_model.keras`)

## Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Data and Model**:
   - Ensure the BraTS 2020 dataset is in `data/MICCAI_BraTS2020_TrainingData`.
   - Ensure the trained model is at `models/my_model.keras`.

## Running the Web Application

To start the interactive web interface:

```bash
# Option 1: Use the startup script (Recommended)
python start_web_app.py

# Option 2: Run via scripts directory
python scripts/run_web_app.py
```

The application will open in your default browser at `http://127.0.0.1:5000`.

## Training and Evaluation

### Train the Model
```bash
python -m brats.train
```

### Evaluate Performance
```bash
python scripts/evaluate_simple.py
```

### Visualize Predictions
```bash
python scripts/show_predicted_segmentations.py
```

## Notes
- The modules use `brats/config.py` to resolve paths dynamically relative to the project root.
- If you move the project, the paths will automatically adjust.
