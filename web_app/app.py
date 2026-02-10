#!/usr/bin/env python3
"""
Complete Flask app for Brain Tumor Segmentation Web Interface
"""
import os
import sys
import json
import subprocess
import threading
import time

# Set matplotlib backend before any imports to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global model instance
model = None

def load_model():
    """Load the pre-trained model"""
    global model
    try:
        from brats.model import load_pretrained_model
        from brats.config import TRAIN_DATASET_PATH, VAL_DATASET_PATH, OUTPUT_DIR
        model = load_pretrained_model()
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/data', methods=['GET'])
def get_available_data():
    """Get list of available cases"""
    try:
        # Find data directories
        train_root = None
        val_root = None
        
        from brats.config import TRAIN_DATASET_PATH, VAL_DATASET_PATH
        print(f"DEBUG: TRAIN_PATH={TRAIN_DATASET_PATH}")
        print(f"DEBUG: VAL_PATH={VAL_DATASET_PATH}")
        
        for root in [TRAIN_DATASET_PATH, VAL_DATASET_PATH]:
            print(f"DEBUG: Checking {root}")
            if os.path.exists(root):
                print(f"DEBUG: Found {root}")
                if "Training" in root:
                    train_root = root
                else:
                    val_root = root
            else:
                print(f"DEBUG: {root} does not exist")
        
        cases = []
        
        # Add training cases
        if train_root:
            for case in sorted(os.listdir(train_root)):
                if case.startswith("BraTS20_Training_") and os.path.isdir(os.path.join(train_root, case)):
                    cases.append({
                        'id': case,
                        'name': f'Training Case {case.split("_")[-1]}',
                        'type': 'TRAIN',
                        'status': 'Available'
                    })
        
        # Add validation cases
        if val_root:
            for case in sorted(os.listdir(val_root)):
                if case.startswith("BraTS20_Validation_") and os.path.isdir(os.path.join(val_root, case)):
                    cases.append({
                        'id': case,
                        'name': f'Validation Case {case.split("_")[-1]}',
                        'type': 'VAL',
                        'status': 'Available'
                    })
        
        print(f"DEBUG: Returning {len(cases)} cases")
        return jsonify({
            'success': True,
            'cases': cases
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/segmentation', methods=['POST'])
def run_segmentation():
    """Run segmentation on selected cases"""
    try:
        data = request.get_json()
        cases = data.get('cases', [])
        styles = data.get('styles', [])
        slices = data.get('slices', '50,60,70,80')
        
        if not cases:
            return jsonify({
                'success': False,
                'error': 'No cases selected'
            }), 400
        
        if not styles:
            return jsonify({
                'success': False,
                'error': 'No styles selected'
            }), 400
        
        # Convert case indices to case IDs
        available_cases = get_available_cases()
        case_ids = []
        for case_index in cases:
            if 0 <= case_index < len(available_cases):
                case_ids.append(available_cases[case_index]['id'])
        
        if not case_ids:
            return jsonify({
                'success': False,
                'error': 'Invalid case selection'
            }), 400
        
        # Run segmentation
        results = run_segmentation_batch(case_ids, styles, slices)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_available_cases():
    """Get list of available cases"""
    cases = []
    
    # Find data directories
    from brats.config import TRAIN_DATASET_PATH, VAL_DATASET_PATH
    for root in [TRAIN_DATASET_PATH, VAL_DATASET_PATH]:
        if os.path.exists(root):
            for case in sorted(os.listdir(root)):
                if (case.startswith("BraTS20_Training_") or case.startswith("BraTS20_Validation_")) and os.path.isdir(os.path.join(root, case)):
                    cases.append({
                        'id': case,
                        'name': f'{"Training" if "Training" in case else "Validation"} Case {case.split("_")[-1]}',
                        'type': 'TRAIN' if 'Training' in case else 'VAL',
                        'status': 'Available'
                    })
    
    return cases

def run_segmentation_batch(case_ids, styles, slices):
    """Run segmentation for multiple cases using direct function calls"""
    results = []
    slice_list = [s.strip() for s in slices.split(',') if s.strip()]
    
    # Create outputs directory
    from brats.config import OUTPUT_DIR
    batch_out_dir = os.path.join(OUTPUT_DIR, 'batch')
    os.makedirs(batch_out_dir, exist_ok=True)
    
    # Get the project root and add to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_app_dir = os.path.dirname(os.path.abspath(__file__))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if web_app_dir not in sys.path:
        sys.path.insert(0, web_app_dir)
    
    # Import the direct segmentation function
    try:
        from web_segmentation_direct import run_segmentation_direct
        print("✅ Successfully imported web_segmentation_direct")
    except ImportError as e:
        print(f"❌ Failed to import web_segmentation_direct: {e}")
        print(f"   sys.path: {sys.path[:3]}")
        # Return error responses with proper structure
        error_results = []
        for case_id in case_ids:
            for style in styles:
                for slice_idx in slice_list:
                    error_results.append({
                        'case': case_id,
                        'style': style,
                        'slice': slice_idx,
                        'error': f'Failed to import segmentation module: {str(e)}',
                        'status': 'failed'
                    })
        return error_results
    
    for case_id in case_ids:
        for style in styles:
            for slice_idx in slice_list:
                try:
                    print(f"Running segmentation for {case_id} - {style} - slice {slice_idx}")
                    
                    # Call the direct segmentation function
                    success = run_segmentation_direct(case_id, style, int(slice_idx))
                    
                    if success:
                        # Check if image was created
                        image_path = os.path.join(OUTPUT_DIR, 'batch', f'{case_id}_s{slice_idx}_{style}.png')
                        if os.path.exists(image_path):
                            result_data = {
                                'case': case_id,
                                'style': style,
                                'slice': slice_idx,
                                'image': image_path,
                                'status': 'completed'
                            }
                            print(f"✅ Image created: {image_path}")
                        else:
                            result_data = {
                                'case': case_id,
                                'style': style,
                                'slice': slice_idx,
                                'error': 'Image not generated',
                                'status': 'failed'
                            }
                            print(f"❌ Image not found: {image_path}")
                    else:
                        result_data = {
                            'case': case_id,
                            'style': style,
                            'slice': slice_idx,
                            'error': 'Segmentation function returned False',
                            'status': 'failed'
                        }
                        print(f"❌ Segmentation failed for {case_id}")
                    
                    results.append(result_data)
                    
                except Exception as e:
                    result_data = {
                        'case': case_id,
                        'style': style,
                        'slice': slice_idx,
                        'error': str(e),
                        'status': 'failed'
                    }
                    results.append(result_data)
                    print(f"❌ Error for {case_id} {style} slice {slice_idx}: {e}")
    
    return results

def get_case_index(case_id):
    """Get the index of a case in the available cases list"""
    available_cases = get_available_cases()
    for i, case in enumerate(available_cases):
        if case['id'] == case_id:
            return i
    return None

@app.route('/api/image/<path:filename>')
def get_image(filename):
    """Serve generated images"""
    try:
        from brats.config import OUTPUT_DIR
        batch_dir = os.path.join(OUTPUT_DIR, 'batch')
        image_path = os.path.join(batch_dir, filename)
        
        # Security check - ensure the file is in the batch directory
        if not os.path.abspath(image_path).startswith(os.path.abspath(batch_dir)):
            return jsonify({'error': 'Invalid file path'}), 403
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        print(f"✅ Serving image: {image_path}")
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        print(f"❌ Error serving image {filename}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🧠 Brain Tumor Segmentation Web App")
    print("=" * 50)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        exit(1)
    
    # Create outputs directory
    from brats.config import OUTPUT_DIR
    os.makedirs(os.path.join(OUTPUT_DIR, 'batch'), exist_ok=True)
    
    print("🚀 Starting server on http://127.0.0.1:5000")
    print("📱 Open your web browser and navigate to the URL above")
    print("=" * 50)
    
    # Run Flask app
    app.run(host='127.0.0.1', port=5000, debug=False)
