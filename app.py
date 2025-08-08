# app.py

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Import our updated functions from the new handler
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
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # This endpoint is fine, no changes needed here.
    if 'crop_image' not in request.files or 'insect_image' not in request.files:
        return jsonify({"error": "Both crop and insect images are required."}), 400
    crop_file = request.files['crop_image']
    insect_file = request.files['insect_image']
    if crop_file.filename == '' or insect_file.filename == '':
        return jsonify({"error": "Please select both files."}), 400

    crop_filename = secure_filename(crop_file.filename)
    insect_filename = secure_filename(insect_file.filename)
    crop_image_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
    insect_image_path = os.path.join(app.config['UPLOAD_FOLDER'], insect_filename)
    crop_file.save(crop_image_path)
    insect_file.save(insect_image_path)
    
    yolo_disease_result = predict_disease_yolo(crop_image_path)
    yolo_insect_result = predict_insect_yolo(insect_image_path)
    questions = get_symptom_questions()
    
    response_data = {
        "image_name": crop_filename, 
        "yolo_disease": yolo_disease_result,
        "yolo_insect": yolo_insect_result,
        "questions": questions
    }
    return jsonify(response_data)

@app.route('/fuse', methods=['POST'])
def fuse_all_predictions():
    data = request.json
    image_name = data.get('image_name')
    yolo_disease = data.get('yolo_disease')
    yolo_insect = data.get('yolo_insect')
    answers = data.get('answers')

    # This call now works because the new analyze_symptoms_tabnet returns two floats
    tabnet_disease_prob, tabnet_insect_prob = analyze_symptoms_tabnet(answers)

    # The rest of this endpoint correctly passes the data to the fusion model
    final_output = fuse_predictions(
        image_name,
        yolo_disease,
        tabnet_disease_prob,
        yolo_insect,
        tabnet_insect_prob
    )
    
    # This returns the final JSON to the website, which will look like your image
    return jsonify(final_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)