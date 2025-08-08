# app.py

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import sys

# Import our updated functions
from model_handler import (
    predict_disease_yolo,
    predict_insect_yolo,
    get_symptom_questions,
    analyze_symptoms_tabnet
)

from fusion import fuse_predictions
# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
# Create a folder for user uploads if it doesn't exist
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def home():
    """Serves the main HTML page from the 'templates' folder."""
    return render_template('index.html')


# In your app.py file

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Endpoint for Step 1: Receives a crop image and an insect image,
    runs the correct YOLO model on each, and returns initial results plus questions.
    """
    print("\n--- /analyze endpoint was hit ---")

    # --- UPDATED LOGIC TO HANDLE TWO FILES ---
    if 'crop_image' not in request.files or 'insect_image' not in request.files:
        return jsonify({"error": "Both crop and insect images are required."}), 400

    crop_file = request.files['crop_image']
    insect_file = request.files['insect_image']

    if crop_file.filename == '' or insect_file.filename == '':
        return jsonify({"error": "Please select both files."}), 400

    # --- Process each file with its specific model ---
    if crop_file and insect_file:
        crop_filename = secure_filename(crop_file.filename)
        insect_filename = secure_filename(insect_file.filename)

        crop_image_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
        insect_image_path = os.path.join(app.config['UPLOAD_FOLDER'], insect_filename)

        crop_file.save(crop_image_path)
        insect_file.save(insect_image_path)
        print(f"DEBUG: Saved crop image to {crop_image_path}")
        print(f"DEBUG: Saved insect image to {insect_image_path}")

        # Run each model on its corresponding image
        print("DEBUG: Starting model predictions...")
        yolo_disease_result = predict_disease_yolo(crop_image_path)
        yolo_insect_result = predict_insect_yolo(insect_image_path)
        print("DEBUG: Model predictions finished.")
        
        # Get questions for the next step (this part doesn't change)
        questions = get_symptom_questions()

        response_data = {
            # We'll use the crop image name for reference in the next step
            "image_name": crop_filename, 
            "yolo_disease": yolo_disease_result,
            "yolo_insect": yolo_insect_result,
            "questions": questions
        }
        print("DEBUG: Successfully prepared response data. Sending to browser.")
        return jsonify(response_data)

    return jsonify({"error": "An unknown error occurred with file processing."}), 500


@app.route('/fuse', methods=['POST'])
def fuse_all_predictions():
    """
    Endpoint for Step 2: Receive symptom answers, run TabNet & Fusion, return final diagnosis.
    """
    data = request.json
    
    # Get all necessary data from the frontend
    image_name = data.get('image_name')
    yolo_disease = data.get('yolo_disease')
    yolo_insect = data.get('yolo_insect')
    answers = data.get('answers')

    # Run symptom analysis with TabNet
    tabnet_disease, tabnet_insect = analyze_symptoms_tabnet(answers)

    # Run the final fusion logic
    final_output = fuse_predictions(
        image_name,
        yolo_disease,
        tabnet_disease,
        yolo_insect,
        tabnet_insect
    )
    
    return jsonify(final_output)


if __name__ == '__main__':
    # Use Gunicorn for production instead of app.run()
    app.run(host='0.0.0.0', port=5000, debug=True) # Use debug=True only for development