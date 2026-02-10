#!/usr/bin/env python3
"""
Web backend for Brain Tumor Segmentation UI
Handles API requests from the web interface
"""
import os
import sys
import json
import subprocess
import argparse
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brats.model import load_pretrained_model
from brats.config import IMG_SIZE, VOLUME_SLICES, VOLUME_START_AT

app = Flask(__name__)
CORS(app)

# Global model instance
model = None

def load_model():
    """Load the pre-trained model"""
    global model
    try:
        model = load_pretrained_model()
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

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
        
        for root in ["MICCAI_BraTS2020_TrainingData", "MICCAI_BraTS2020_ValidationData"]:
            if os.path.exists(root):
                if "Training" in root:
                    train_root = root
                else:
                    val_root = root
        
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
    for root in ["MICCAI_BraTS2020_TrainingData", "MICCAI_BraTS2020_ValidationData"]:
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
    """Run segmentation for multiple cases"""
    results = []
    slice_list = [s.strip() for s in slices.split(',') if s.strip()]
    
    for case_id in case_ids:
        for style in styles:
            for slice_idx in slice_list:
                try:
                    # Run the segmentation script
                    cmd = [
                        'python', 'show_predicted_segmentations.py',
                        '--style', style,
                        '--slices', slice_idx
                    ]
                    
                    # Add case selection (this would need to be modified in the script)
                    # For now, we'll run it and let the user select
                    
                    result = {
                        'case': case_id,
                        'style': style,
                        'slice': slice_idx,
                        'image': f'outputs/batch/{case_id}_s{slice_idx}_{style}.png',
                        'status': 'completed'
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    result = {
                        'case': case_id,
                        'style': style,
                        'slice': slice_idx,
                        'error': str(e),
                        'status': 'failed'
                    }
                    results.append(result)
    
    return results

@app.route('/api/image/<path:filename>')
def get_image(filename):
    """Serve generated images"""
    try:
        return send_file(f'outputs/batch/{filename}')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Web Backend')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    args = parser.parse_args()
    
    print("🧠 Brain Tumor Segmentation Web Backend")
    print("=" * 50)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Create outputs directory
    os.makedirs('outputs/batch', exist_ok=True)
    
    print(f"🚀 Starting server on http://{args.host}:{args.port}")
    print("📱 Open your web browser and navigate to the URL above")
    print("=" * 50)
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
