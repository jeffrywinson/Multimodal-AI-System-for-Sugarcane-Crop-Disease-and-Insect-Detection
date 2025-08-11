# model_handler.py (Definitive Final Version)
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from ultralytics import YOLO

# ==============================================================================
# 1. FINAL, CORRECT QUESTIONS FOR ROUND 2
# ==============================================================================
insect_rules = [
    ("Is the crop ≤ 120 days old (i.e., within first 4 months)?", "ESB_positive"),
    ("Did the damage start after 4 months from planting?", "INB_positive"),
    ("Did you first notice the damage between March and June?", "ESB_positive"),
    ("Did you first notice the damage between June and December?", "INB_positive"),
    ("Did attacks start very soon after planting (within about 15 days)?", "ESB_positive"),
    ("Did the damage stop appearing after 4 months from planting?", "ESB_positive"),
    ("Is the peak damage appearing around 7–9 months after planting?", "INB_positive"),
    ("Does the damage seem to start from the lower part of the stalk?", "ESB_positive"),
    ("Does the damage seem to start from the upper part of the stalk?", "INB_positive"),
    ("Are bore holes within 15 cm from the soil?", "ESB_positive"),
    ("Are bore holes on the upper internodes?", "INB_positive"),
    ("Are some stalk internodes malformed or constricted?", "INB_positive"),
    ("Do you see small aerial roots appearing on the stalks above the ground?", "INB_positive"),
    ("When you pull the dead central shoot, does it come out easily?", "ESB_positive"),
    ("Does the pulled shoot have a foul or bad smell?", "ESB_positive"),
    ("Is the pulled shoot difficult to remove?", "INB_positive"),
    ("When removed, is there no bad smell from the shoot?", "INB_positive"),
    ("Are insect eggs present on the underside of the lower leaves?", "ESB_positive"),
    ("Are insect eggs present on the underside of the top leaves?", "INB_positive"),
    ("Are the eggs flat, white, and smaller than 1 mm?", "ESB_positive"),
    ("Does the larva have 5 visible stripes along its body?", "ESB_positive"),
    ("Does the larva have only 4 visible stripes along its body?", "INB_positive"),
    ("Is fresh powder-like excreta seen near the base of the stalk?", "ESB_positive"),
    ("Is fresh excreta seen on the upper internodes?", "INB_positive"),
    ("In the early stages, do you see only green leaf scraping without bore holes?", "ESB_positive"),
    ("Has the damage occurred only after internodes have fully developed?", "INB_positive"),
    ("Along with dead shoots, do you see bunchy or abnormal top growth?", "INB_positive"),
    ("Did you apply a high dose of nitrogen/urea before the damage started?", "general_borer_positive"),
    ("Was trash mulching done early in the crop stage?", "ESB_management_positive"),
    ("Was earthing-up done to cover the lower stalk area?", "ESB_management_positive")
]
dead_heart_rules = [
    ("Have you seen the central growing point of the stalk damaged or dead?", "dead_heart_confirmation_positive"),
    ("Is the dead central shoot straw-coloured?", "dead_heart_confirmation_positive"),
    ("After pulling the dead-heart, do you find maggots that look like secondary feeders (not primary borers)?", "dead_heart_confirmation_positive"),
    ("After pulling the dead-heart, do you find fresh live larvae inside the affected stem?", "dead_heart_confirmation_positive"),
    ("Is most of the visible damage inside the stem rather than outside?", "dead_heart_confirmation_positive"),
    ("Have you noticed insect attack when the leaves are still developing and soft?", "dead_heart_confirmation_positive"),
    ("Was the crop planted within the last 15 days?", "dead_heart_timing_negative"),
    ("Have you never seen moths flying during daytime?", "dead_heart_cause_neutral"),
    ("Have you observed mating or egg-laying activity mostly at night?", "dead_heart_cause_neutral"),
    ("Were any biological control insects released in the field?", "dead_heart_cause_neutral"),
    ("Have you seen fully grown moths that are straw to light brown in colour?", "dead_heart_cause_positive"),
    ("Is the central shoot of young plants dry, brown, or straw-colored?", "dead_heart_confirmation_positive"),
    ("Does the central shoot come out easily when pulled gently?", "dead_heart_confirmation_positive"),
    ("Does the pulled shoot emit a foul or rotten odor?", "dead_heart_confirmation_positive"),
    ("Are leaves around the central shoot yellowing, wilting, or drying?", "dead_heart_confirmation_positive"),
    ("Do patches of plants show multiple stalks with dried or dead centers?", "dead_heart_confirmation_positive"),
    ("Has the percentage of dead hearts increased after recent rains or waterlogging?", "dead_heart_confirmation_positive"),
    ("Are affected plants stunted or shorter than surrounding healthy plants?", "dead_heart_confirmation_positive"),
    ("Do affected plants fail to produce new green shoots or leaves?", "dead_heart_confirmation_positive"),
    ("Are there soft, hollow, or tunnel-like areas inside the affected stalks?", "dead_heart_confirmation_positive"),
    ("Have you seen bunchy or abnormal growth at the top of affected stalks?", "dead_heart_confirmation_positive"),
    ("Are soil moisture and drainage poor in areas where dead hearts appear?", "dead_heart_confirmation_positive"),
    ("Are there no dry central shoots in the field?", "dead_heart_confirmation_negative"),
    ("Is plant height uniform and normal throughout the field?", "dead_heart_confirmation_negative"),
    ("When pulling the central shoot, is it firmly attached without coming out easily?", "dead_heart_confirmation_negative"),
    ("Does the shoot base smell fresh with no rotting or foul odor?", "dead_heart_confirmation_negative"),
    ("Are leaves healthy, green, and not wilting near the central shoot?", "dead_heart_confirmation_negative"),
    ("Do you have no patches with multiple dead or dried shoots?", "dead_heart_confirmation_negative"),
    ("Have symptoms decreased after improved irrigation or fertilization?", "dead_heart_confirmation_negative"),
    ("Is there no recurrence of dead heart symptoms from previous seasons?", "dead_heart_confirmation_negative")
]


# --- MODEL LOADING ---
print("--- Loading all models into memory... ---")
TABNET_DISEASE_MODEL = TabNetClassifier(); TABNET_DISEASE_MODEL.load_model('./tabnet/tabnet_disease_model.zip')
TABNET_INSECT_MODEL = TabNetClassifier(); TABNET_INSECT_MODEL.load_model('./tabnet/tabnet_insect_model.zip')
YOLO_DISEASE_MODEL = YOLO("./YOLOv8s-seg/best.pt")
YOLO_INSECT_MODEL = YOLO(r"D:\finalllll\Multimodal-AI-System-for-Sugarcane-Crop-Disease-and-Insect-Detection\YOLOv8s\larva\best (1).pt")
print("--- All models loaded successfully. ---")


# --- PREDICTION FUNCTIONS ---

# ==============================================================================
# 2. DEFINE THE PREDICTION FUNCTIONS (FINAL VERSION)
# ==============================================================================

def get_symptom_questions():
    """
    Returns a structured dictionary of questions with unique, language-independent keys.
    THIS IS THE KEY FIX FOR THE TRANSLATION BUG.
    """
    disease_question_list = [{"key": f"dh_q{i+1}", "text": rule[0]} for i, rule in enumerate(dead_heart_rules)]
    insect_question_list = [{"key": f"in_q{i+1}", "text": rule[0]} for i, rule in enumerate(insect_rules)]
    
    return {
        "disease_questions": disease_question_list,
        "insect_questions": insect_question_list
    }

def predict_disease_yolo(image_path):
    """Returns disease area percentage."""
    results = YOLO_DISEASE_MODEL.predict(image_path, verbose=False)
    if results[0].masks:
        h, w = results[0].orig_shape; image_area = h * w
        total_disease_area = sum(mask.data.sum() for mask in results[0].masks)
        return (total_disease_area / image_area).item()
    return 0.0

def predict_insect_yolo(image_path):
    """Returns the number of insects found."""
    results = YOLO_INSECT_MODEL.predict(image_path, verbose=False)
    return float(len(results[0].boxes))

def analyze_symptoms_tabnet(disease_answers, insect_answers):
    """Takes answer lists and returns TabNet model predictions."""
    disease_answers = disease_answers or []
    insect_answers = insect_answers or []
    
    disease_input = np.array([[1 if ans and ans.lower() == 'yes' else 0 for ans in disease_answers]]).reshape(1, -1)
    if disease_input.shape[1] == 0: disease_input = np.zeros((1, 30))
    disease_prob = TABNET_DISEASE_MODEL.predict_proba(disease_input)[0][1]

    insect_input = np.array([[1 if ans and ans.lower() == 'yes' else 0 for ans in insect_answers]]).reshape(1, -1)
    if insect_input.shape[1] == 0: insect_input = np.zeros((1, 30))
    pred_index = TABNET_INSECT_MODEL.predict(insect_input)[0]
    class_map = {0: "Early Shoot Borer", 1: "Internode Borer", 2: "No Insect"}
    insect_class = class_map.get(pred_index, "No Insect")
    
    return disease_prob, insect_class