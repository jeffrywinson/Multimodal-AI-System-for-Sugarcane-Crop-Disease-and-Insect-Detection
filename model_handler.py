# model_handler.py (FINAL VERSION)

import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from ultralytics import YOLO

# --- Step 1: Copy your exact question lists here ---
disease_rules = [
    ("Is there a yellow halo around the spots?", "Positive"),
    ("Are the leaf spots circular with concentric rings?", "Positive"),
    ("Does the disease begin on the lower leaves?", "Positive"),
    ("Are the lesions expanding over time?", "Positive"),
    ("Is the center of the spot dry and brown?", "Positive"),
    ("Are multiple spots merging to form large blotches?", "Positive"),
    ("Does the leaf show signs of early yellowing?", "Positive"),
    ("Are stems or fruits also affected?", "Positive"),
    ("Are the affected leaves wilting?", "Positive"),
    ("Is the infection spreading upward on the plant?", "Positive"),
    ("Are concentric rings visible clearly on the leaves?", "Positive"),
    ("Is there any rotting seen on fruit?", "Positive"),
    ("Are the leaf margins turning brown?", "Positive"),
    ("Is the plant under moisture stress?", "Positive"),
    ("Is the disease more active during rainy days?", "Positive"),
    ("Are nearby tomato plants also showing similar symptoms?", "Positive"),
    ("Is there any black moldy growth on the lesion?", "Positive"),
    ("Does the disease affect the whole plant?", "Positive"),
    ("Is the spot size more than 5mm in diameter?", "Positive"),
    ("Are the lesions visible on both sides of the leaf?", "Positive"),
    ("Is the infection found only on mature leaves?", "Positive"),
    ("Are the leaf veins visible through the lesion?", "Positive"),
    ("Is the damage uniform across the field?", "Positive"),
    ("Was there previous history of Early Blight in this field?", "Positive"),
    ("Is the farmer using resistant tomato varieties?", "Negative"),
    ("Was any fungicide recently applied?", "Negative"),
    ("Was there poor air circulation in the field?", "Positive"),
    ("Was the field irrigated from overhead sprinklers?", "Positive"),
    ("Are pruning and sanitation practices followed?", "Negative"),
    ("Is there any other crop in the field showing similar spots?", "Positive")
]
insect_rules = [
    ("Is the pest in the image an armyworm?", "Positive"),
    ("Is the armyworm green in color?", "Positive"),
    ("Is the armyworm brown in color?", "Positive"),
    ("Is the armyworm found on the leaf top?", "Positive"),
    ("Is the armyworm found on the underside of the leaf?", "Positive"),
    ("Is the armyworm present on the stem?", "Positive"),
    ("Is the armyworm feeding on the crop?", "Positive"),
    ("Are visible bite marks present on the leaf?", "Positive"),
    ("Are there multiple armyworms in the image?", "Positive"),
    ("Is any frass (armyworm waste) visible near the pest?", "Positive"),
    ("Are eggs visible near the armyworm?", "Positive"),
    ("Are larvae of the armyworm visible?", "Positive"),
    ("Has the crop been attacked by armyworm in previous seasons?", "Positive"),
    ("Was pesticide recently applied to this crop area?", "Negative"),
    ("Is the armyworm population increasing?", "Positive"),
    ("Is the armyworm active during daylight hours?", "Positive"),
    ("Is the armyworm mostly active during night?", "Positive"),
    ("Is the leaf portion of the plant affected?", "Positive"),
    ("Is the stem portion of the plant affected?", "Positive"),
    ("Is the damage restricted to a small part of the crop?", "Negative"),
    ("Are nearby plants also showing signs of armyworm infestation?", "Positive"),
    ("Is the armyworm moving actively?", "Positive"),
    ("Are there signs of curled leaves due to feeding?", "Positive"),
    ("Has the armyworm damaged more than one section of the same plant?", "Positive"),
    ("Is there visible discoloration of the crop due to pest feeding?", "Positive"),
    ("Does the armyworm show striping or lines on its body?", "Positive"),
    ("Is the length of the armyworm greater than 20 mm?", "Positive"),
    ("Are any dead armyworms seen in the area (possibly due to pesticide)?", "Negative"),
    ("Is any chewing sound audible during the inspection?", "Positive"),
    ("Has any farmer nearby reported armyworm infestation in the last week?", "Positive")
]

# --- Step 2: Automatically create the feature names your model was trained on ---
DISEASE_FEATURES = [f'q_{i+1}' for i in range(len(disease_rules))]
INSECT_FEATURES = [f'q_{i+1}' for i in range(len(insect_rules))]

# --- MODEL LOADING ---
# (This section is the same as before, no changes needed here)
print("--- Loading all models into memory... ---")
TABNET_DISEASE_MODEL = TabNetClassifier()
TABNET_DISEASE_MODEL.load_model('./tabnet/tabnet_disease_model.zip')
TABNET_INSECT_MODEL = TabNetClassifier()
TABNET_INSECT_MODEL.load_model('./tabnet/tabnet_insect_model.zip')
YOLO_DISEASE_MODEL = YOLO("YOLOv8s-seg/best.pt")
YOLO_INSECT_MODEL = YOLO("YOLOv8s/yolov8s_insect_detection_best.pt")
print("--- All models loaded successfully. ---")


# --- PREDICTION FUNCTIONS ---

def get_symptom_questions():
    """
    Returns a structured list of questions for the frontend.
    This is the key change to fix the 'undefined' error.
    """
    disease_question_list = [{"id": f"q_{i+1}", "text": rule[0]} for i, rule in enumerate(disease_rules)]
    insect_question_list = [{"id": f"q_{i+1}", "text": rule[0]} for i, rule in enumerate(insect_rules)]

    return {
        "disease_questions": disease_question_list,
        "insect_questions": insect_question_list
    }

# The rest of your functions are now correct and don't need changes.
def predict_disease_yolo(image_path):
    results = YOLO_DISEASE_MODEL.predict(image_path, verbose=False)
    return "Present" if len(results[0].boxes) > 0 else "Not Present"

def predict_insect_yolo(image_path):
    results = YOLO_INSECT_MODEL.predict(image_path, verbose=False)
    return "Present" if len(results[0].boxes) > 0 else "Not Present"

def analyze_symptoms_tabnet(answers):
    # This function now works perfectly because the 'answers' dictionary from the
    # website will have keys like 'q_1', 'q_2', etc., which match DISEASE_FEATURES.
    disease_input_data = {feature: 1 if answers.get(feature) == 'yes' else 0 for feature in DISEASE_FEATURES}
    disease_df = pd.DataFrame([disease_input_data])[DISEASE_FEATURES]

    insect_input_data = {feature: 1 if answers.get(feature) == 'yes' else 0 for feature in INSECT_FEATURES}
    insect_df = pd.DataFrame([insect_input_data])[INSECT_FEATURES]
    
    disease_probs = TABNET_DISEASE_MODEL.predict_proba(disease_df.to_numpy())
    insect_probs = TABNET_INSECT_MODEL.predict_proba(insect_df.to_numpy())

    tabnet_disease_prob = disease_probs[0][1]
    tabnet_insect_prob = insect_probs[0][1]
    
    return tabnet_disease_prob, tabnet_insect_prob