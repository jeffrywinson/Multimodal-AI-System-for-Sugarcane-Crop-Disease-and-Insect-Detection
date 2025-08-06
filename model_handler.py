# model_handler.py

import os
import pandas as pd
from ultralytics import YOLO
from pytorch_tabnet.tab_model import TabNetClassifier

# --- Define Model and Feature Information ---
# Paths remain the same
DISEASE_MODEL_PATH = 'YOLOv8s-seg/best.pt'
INSECT_MODEL_PATH = 'YOLOv8s/yolov8s_insect_detection_best.pt'
TABNET_DISEASE_PATH = 'tabnet/tabnet_disease_model.zip'
TABNET_INSECT_PATH = 'tabnet/tabnet_insect_model.zip'

DISEASE_FEATURES = ['q_1','q_2','q_3','q_4','q_5','q_6','q_7','q_8','q_9','q_10','q_11','q_12','q_13','q_14','q_15','q_16','q_17','q_18','q_19','q_20','q_21','q_22','q_23','q_24','q_25','q_26','q_27','q_28','q_29','q_30'] 
INSECT_FEATURES = ['q_1','q_2','q_3','q_4','q_5','q_6','q_7','q_8','q_9','q_10','q_11','q_12','q_13','q_14','q_15','q_16','q_17','q_18','q_19','q_20','q_21','q_22','q_23','q_24','q_25','q_26','q_27','q_28','q_29','q_30']


# --- 2. YOUR ACTION: DEFINE SEPARATE QUESTION MAPS ---
# Map your disease column names to user-friendly questions
DISEASE_QUESTIONS_MAP = {
    'd_q1': "Is there a yellow halo around the spots?",
    'd_q2': "Are the leaf spots circular with concentric rings?",
    'd_q3': "Does the disease begin on the lower leaves?",
    'd_q4': "Are the lesions expanding over time?",
    'd_q5': "Is the center of the spot dry and brown?",
    'd_q6': "Are multiple spots merging to form large blotches?",
    'd_q7': "Does the leaf show signs of early yellowing?",
    'd_q8': "Are stems or fruits also affected?",
    'd_q9': "Are the affected leaves wilting?",
    'd_q10': "Is the infection spreading upward on the plant?",
    'd_q11': "Are concentric rings visible clearly on the leaves?",
    'd_q12': "Is there any rotting seen on fruit?",
    'd_q13': "Are the leaf margins turning brown?",
    'd_q14': "Is the plant under moisture stress?",
    'd_q15': "Is the disease more active during rainy days?",
    'd_q16': "Are nearby tomato plants also showing similar symptoms?",
    'd_q17': "Is there any black moldy growth on the lesion?",
    'd_q18': "Does the disease affect the whole plant?",
    'd_q19': "Is the spot size more than 5mm in diameter?",
    'd_q20': "Are the lesions visible on both sides of the leaf?",
    'd_q21': "Is the infection found only on mature leaves?",
    'd_q22': "Are the leaf veins visible through the lesion?",
    'd_q23': "Is the damage uniform across the field?",
    'd_q24': "Was there previous history of Early Blight in this field?",
    'd_q25': "Is the farmer using resistant tomato varieties?",
    'd_q26': "Was any fungicide recently applied?",
    'd_q27': "Was there poor air circulation in the field?",
    'd_q28': "Was the field irrigated from overhead sprinklers?",
    'd_q29': "Are pruning and sanitation practices followed?",
    'd_q30': "Is there any other crop in the field showing similar spots?"
}

# Map your insect column names to user-friendly questions
INSECT_QUESTIONS_MAP = {
    'd_q1': "Is the pest in the image an armyworm?",
    'd_q2': "Is the armyworm green in color?",
    'd_q3': "Is the armyworm brown in color?",
    'd_q4': "Is the armyworm found on the leaf top?",
    'd_q5': "Is the armyworm found on the underside of the leaf?",
    'd_q6': "Is the armyworm present on the stem?",
    'd_q7': "Is the armyworm feeding on the crop?",
    'd_q8': "Are visible bite marks present on the leaf?",
    'd_q9': "Are there multiple armyworms in the image?",
    'd_q10': "Is any frass (armyworm waste) visible near the pest?",
    'd_q11': "Are eggs visible near the armyworm?",
    'd_q12': "Are larvae of the armyworm visible?",
    'd_q13': "Has the crop been attacked by armyworm in previous seasons?",
    'd_q14': "Was pesticide recently applied to this crop area?",
    'd_q15': "Is the armyworm population increasing?",
    'd_q16': "Is the armyworm active during daylight hours?",
    'd_q17': "Is the armyworm mostly active during night?",
    'd_q18': "Is the leaf portion of the plant affected?",
    'd_q19': "Is the stem portion of the plant affected?",
    'd_q20': "Is the damage restricted to a small part of the crop?",
    'd_q21': "Are nearby plants also showing signs of armyworm infestation?",
    'd_q22': "Is the armyworm moving actively?",
    'd_q23': "Are there signs of curled leaves due to feeding?",
    'd_q24': "Has the armyworm damaged more than one section of the same plant?",
    'd_q25': "Is there visible discoloration of the crop due to pest feeding?",
    'd_q26': "Does the armyworm show striping or lines on its body?",
    'd_q27': "Is the length of the armyworm greater than 20 mm?",
    'd_q28': "Are any dead armyworms seen in the area (possibly due to pesticide)?",
    'd_q29': "Is any chewing sound audible during the inspection?",
    'd_q30': "Has any farmer nearby reported armyworm infestation in the last week?"

}

# --- Pre-load all models on startup ---
print("Loading models, this may take a moment...")
DISEASE_YOLO_MODEL = YOLO(DISEASE_MODEL_PATH)
INSECT_YOLO_MODEL = YOLO(INSECT_MODEL_PATH)

TABNET_DISEASE_MODEL = TabNetClassifier()
TABNET_DISEASE_MODEL.load_model(TABNET_DISEASE_PATH)

TABNET_INSECT_MODEL = TabNetClassifier()
TABNET_INSECT_MODEL.load_model(TABNET_INSECT_PATH)
print("All models loaded successfully.")


# --- YOLO Prediction Functions (No changes here) ---
def predict_disease_yolo(image_path):
    results = DISEASE_YOLO_MODEL.predict(image_path, verbose=False)
    if results[0].masks is not None and len(results[0].masks) > 0:
        return "Present"
    else:
        return "Not Present"

def predict_insect_yolo(image_path):
    results = INSECT_YOLO_MODEL.predict(image_path, verbose=False)
    if len(results[0].boxes) > 0:
        return "Present"
    else:
        return "Not Present"

# --- TabNet Question and Prediction Functions (CORRECTED LOGIC) ---
def get_symptom_questions():
    """
    Returns two separate lists of questions, one for disease and one for insects.
    """
    disease_questions = [{"id": feature_id, "text": text} for feature_id, text in DISEASE_QUESTIONS_MAP.items()]
    insect_questions = [{"id": feature_id, "text": text} for feature_id, text in INSECT_QUESTIONS_MAP.items()]
    
    return {
        "disease_questions": disease_questions,
        "insect_questions": insect_questions
    }

def analyze_symptoms_tabnet(answers):
    """
    Runs TabNet predictions on two separate sets of features.
    """
    # Create the DataFrame for the DISEASE model
    disease_input_data = {feature: 1 if answers.get(feature) == 'yes' else 0 for feature in DISEASE_FEATURES}
    disease_df = pd.DataFrame([disease_input_data])[DISEASE_FEATURES]

    # Create the DataFrame for the INSECT model
    insect_input_data = {feature: 1 if answers.get(feature) == 'yes' else 0 for feature in INSECT_FEATURES}
    insect_df = pd.DataFrame([insect_input_data])[INSECT_FEATURES]
    
    # Predict using the respective models
    disease_pred = TABNET_DISEASE_MODEL.predict(disease_df.to_numpy())[0]
    insect_pred = TABNET_INSECT_MODEL.predict(insect_df.to_numpy())[0]

    # Convert predictions to string output
    tabnet_disease_result = "Present" if disease_pred == 1 else "Not Present"
    tabnet_insect_result = "Present" if insect_pred == 1 else "Not Present"
    
    return tabnet_disease_result, tabnet_insect_result