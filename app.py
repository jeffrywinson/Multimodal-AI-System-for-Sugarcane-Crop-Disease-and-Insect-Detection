# app.py (Updated with Recommendations and Error Handling)

import os
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json

from model_handler import (
    predict_disease_yolo,
    predict_insect_yolo,
    get_symptom_questions,
    analyze_symptoms_tabnet
)
# This is the correct import for your final fusion model
from fusion import get_fused_prediction_rules

# --- Solutions Database ---
# A simple dictionary to hold recommendations for each diagnosis.
solutions_db = {
    "Dead Heart Present": {
        "en": "Remove and destroy affected shoots. Apply recommended insecticides like Cartap Hydrochloride. Ensure proper water drainage.",
        "ta": "பாதிக்கப்பட்ட தளிர்களை அகற்றி அழிக்கவும். கார்டாப் ஹைட்ரோகுளோரைடு போன்ற பரிந்துரைக்கப்பட்ட பூச்சிக்கொல்லிகளைப் பயன்படுத்தவும். சரியான நீர் வடிகால் உறுதி செய்யவும்.",
        "hi": "प्रभावित अंकुरों को हटाकर नष्ट कर दें। कार्टैप हाइड्रोक्लोराइड जैसे अनुशंसित कीटनाशकों का प्रयोग करें। उचित जल निकासी सुनिश्चित करें।"
    },
    "Early Shoot Borer": {
        "en": "Release Trichogramma egg parasitoids as a biological control. Avoid late planting. Apply granules of Chlorantraniliprole.",
        "ta": "உயிரியல் கட்டுப்பாட்டிற்கு டிரைகோகிராமா முட்டை ஒட்டுண்ணிகளை வெளியிடவும். தாமதமாக நடுவதைத் தவிர்க்கவும். குளோரான்ட்ரனிலிப்ரோல் துகள்களைப் பயன்படுத்தவும்.",
        "hi": "जैविक नियंत्रण के रूप में ट्राइकोग्रामा अंडा परजीवी छोड़ें। देर से रोपण से बचें। क्लोरेंट्रानिलिप्रोल के दाने डालें।"
    },
    "Healthy": {
        "en": "Crop appears healthy. Continue regular monitoring and good agricultural practices.",
        "ta": "பயிர் ஆரோக்கியமாகத் தெரிகிறது. வழக்கமான கண்காணிப்பு மற்றும் நல்ல விவசாய நடைமுறைகளைத் தொடரவும்.",
        "hi": "फसल स्वस्थ दिख रही है। नियमित निगरानी और अच्छी कृषि पद्धतियों को जारी रखें।"
    }
    # Add other potential diagnoses and their solutions here
}


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
    if 'image_file' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    image_file = request.files['image_file']
    if image_file.filename == '':
        return jsonify({"error": "Please select an image file."}), 400
        
    image_filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image_file.save(image_path)

    try:
        yolo_disease_area = predict_disease_yolo(image_path)
        yolo_insect_count = predict_insect_yolo(image_path)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return jsonify({"error": "Failed to analyze the image. It might be corrupted or in an unsupported format."}), 500
    
    all_questions = get_symptom_questions()
    questions_to_send = {}
    image_type = "Analysis Result"

    if yolo_disease_area > 0:
        questions_to_send['disease_questions'] = all_questions['disease_questions']
        image_type = "Image appears to show Dead Heart symptoms."
    if yolo_insect_count > 0:
        questions_to_send['insect_questions'] = all_questions['insect_questions']
        if yolo_disease_area > 0:
             image_type = "Image appears to show both Dead Heart and Insects."
        else:
             image_type = "Image appears to show Insects."
    
    if not questions_to_send:
        # If no issues are detected, still provide the option to answer questions
        questions_to_send = all_questions
        image_type = "No immediate issues detected. Please answer questions for a deeper analysis."

    image_url = url_for('static', filename=f'uploads/{image_filename}')

    response_data = {
        "image_content_type": image_type,
        "image_url": image_url,
        "yolo_disease_output": yolo_disease_area,
        "yolo_insect_output": yolo_insect_count,
        "questions": questions_to_send
    }
    return jsonify(response_data)


@app.route('/fuse', methods=['POST'])
def fuse_all_predictions():
    data = request.json
    yolo_disease_area = data.get('yolo_disease_output')
    yolo_insect_count = data.get('yolo_insect_output')
    disease_answers = data.get('disease_answers')
    insect_answers = data.get('insect_answers')

    tabnet_disease_prob, tabnet_insect_class = analyze_symptoms_tabnet(disease_answers, insect_answers)

    # Use the correct, final fusion function
    final_output = get_fused_prediction_rules(
        yolo_disease_area,
        yolo_insect_count,
        tabnet_disease_prob,
        tabnet_insect_class
    )
    
    # --- Add Recommendations to the final output ---
    disease_key = final_output.get('dead_heart_analysis', {}).get('final_diagnosis', 'Healthy')
    insect_key = final_output.get('insect_analysis', {}).get('final_diagnosis', 'Healthy')
    
    if 'dead_heart_analysis' in final_output:
        final_output['dead_heart_analysis']['recommendation'] = solutions_db.get(disease_key, {})
        
    if 'insect_analysis' in final_output:
        final_output['insect_analysis']['recommendation'] = solutions_db.get(insect_key, {})

    return jsonify(final_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)