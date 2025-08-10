# app.py (Definitive Final Version for Round 2)

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS

from model_handler import (
    predict_disease_yolo,
    predict_insect_yolo,
    get_symptom_questions,
    analyze_symptoms_tabnet
)
# --- This is the only import that changes ---
from fusion import get_fused_prediction_rules
# We are no longer importing get_fused_prediction

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    This endpoint handles the initial image upload.
    It runs the YOLO models and sends back the questions for the next step.
    """
    if 'crop_image' not in request.files:
        return jsonify({"error": "Crop image is required."}), 400
    
    crop_file = request.files['crop_image']
    if crop_file.filename == '':
        return jsonify({"error": "Please select an image file."}), 400
        
    image_filename = secure_filename(crop_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    crop_file.save(image_path)

    # Run both YOLO models on the uploaded image
    yolo_disease_area = predict_disease_yolo(image_path)
    yolo_insect_count = predict_insect_yolo(image_path)
    
    # Get the correctly formatted questions
    questions = get_symptom_questions() # This now returns a dictionary
    
    # Send everything the frontend needs for the next step
    response_data = {
        "yolo_disease_output": yolo_disease_area,
        "yolo_insect_output": yolo_insect_count,
        "questions": questions # Send the dictionary of questions
    }
    return jsonify(response_data)

app.route('/fuse', methods=['POST'])
def fuse_all_predictions():
    """
    This endpoint receives the YOLO results and the user's answers,
    runs the TabNet and the TWO Fusion models, and returns the final diagnosis.
    """
    data = request.json
    yolo_disease_area = data.get('yolo_disease_output')
    yolo_insect_count = data.get('yolo_insect_output')
    disease_answers = data.get('disease_answers')
    insect_answers = data.get('insect_answers')

    # Run TabNet analysis (this function call stays the same)
    tabnet_disease_prob, tabnet_insect_class = analyze_symptoms_tabnet(disease_answers, insect_answers)

    # --- THIS IS THE ONLY FUNCTION CALL THAT CHANGES ---
    # Run the final fusion using the new coordinator function
    final_output = get_fused_prediction_rules(
        yolo_disease_area,
        yolo_insect_count,
        tabnet_disease_prob,
        tabnet_insect_class
    )
    # ----------------------------------------------------
    
    return jsonify(final_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)