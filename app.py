# app.py (Definitive Final Version for Round 2)

import os
from flask import Flask, request, jsonify, render_template, url_for
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
    This endpoint handles a SINGLE image upload.
    It runs BOTH YOLO models, returns the findings, the relevant questions,
    and the URL of the uploaded image for preview.
    """
    if 'image_file' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    image_file = request.files['image_file']
    if image_file.filename == '':
        return jsonify({"error": "Please select an image file."}), 400
        
    image_filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image_file.save(image_path)

    # --- Run BOTH models on the SAME image ---
    yolo_disease_area = predict_disease_yolo(image_path)
    yolo_insect_count = predict_insect_yolo(image_path)
    
    all_questions = get_symptom_questions()
    questions_to_send = {}
    image_type = "Unknown"

    if yolo_disease_area > 0:
        questions_to_send['disease_questions'] = all_questions['disease_questions']
        image_type = "Image contains Dead Heart symptom"

    if yolo_insect_count > 0:
        questions_to_send['insect_questions'] = all_questions['insect_questions']
        if image_type.startswith("Image contains Dead Heart"):
            image_type = "Image contains both Dead Heart and Larva"
        else:
            image_type = "Image contains Larva"
            
    if not questions_to_send:
        image_type = "Neither Dead Heart nor Larva detected"
    
    # MODIFICATION: Create a URL for the saved image
    image_url = url_for('static', filename=f'uploads/{image_filename}')

    response_data = {
        "image_content_type": image_type,
        "image_url": image_url, # Add the URL to the response
        "yolo_disease_output": yolo_disease_area,
        "yolo_insect_output": yolo_insect_count,
        "questions": questions_to_send
    }
    return jsonify(response_data)


@app.route('/fuse', methods=['POST'])
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
