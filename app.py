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


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Endpoint for Step 1: Upload image, run YOLO models, return initial results and questions.
    """
    # --- Start of Debugging Code ---
    print("\n--- /analyze endpoint was hit ---") # Checkpoint 1: Confirms the function is being called.
    
    # This print statement is the most important one. Let's see what's inside request.files.
    print(f"Contents of request.files: {request.files}") # Checkpoint 2: Shows what files the server received.
    # --- End of Debugging Code ---

    if 'file' not in request.files:
        print("DEBUG: 'file' key was NOT found in request.files.") # Checkpoint 3
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("DEBUG: File was found, but its filename is empty.") # Checkpoint 4
        return jsonify({"error": "No selected file"}), 400

    if file:
        print(f"DEBUG: File '{file.filename}' received. Proceeding with analysis.") # Checkpoint 5
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Run initial image analysis with YOLO
        print("DEBUG: Starting YOLO predictions...")
        yolo_disease_result = predict_disease_yolo(image_path)
        yolo_insect_result = predict_insect_yolo(image_path)
        print("DEBUG: YOLO predictions finished.")
        
        # Get questions for the next step
        questions = get_symptom_questions()

        response_data = {
            "image_name": filename,
            "yolo_disease": yolo_disease_result,
            "yolo_insect": yolo_insect_result,
            "questions": questions
        }
        print("DEBUG: Successfully prepared response data. Sending to browser.")
        return jsonify(response_data)

    # This part should ideally not be reached, but is here for completeness.
    return jsonify({"error": "An unknown error occurred"}), 500


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