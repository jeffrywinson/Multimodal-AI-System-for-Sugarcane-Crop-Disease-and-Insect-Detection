# fusion.py (Final Robust Version, modified for web app)

import json 

def fuse_predictions(image_name, yolo_disease, tabnet_disease, yolo_insect, tabnet_insect):
    final_output = {
        "image_name": image_name,
        "disease_analysis": {},
        "insect_analysis": {}
    }

    # --- Fuse Disease Predictions ---
    is_yolo_disease_present = "Present" in yolo_disease
    is_tabnet_disease_present = "Present" in tabnet_disease

    if is_yolo_disease_present and is_tabnet_disease_present:
        disease_diagnosis = "Confirmed: Disease Present (High Confidence)"
    elif is_yolo_disease_present:
        disease_diagnosis = "Possible: Disease Present (Image Only)"
    elif is_tabnet_disease_present:
        disease_diagnosis = "Possible: Disease Present (Symptoms Reported)"
    else:
        disease_diagnosis = "No Disease Detected"
        
    final_output["disease_analysis"] = {
        "image_detection": yolo_disease.strip(),
        "symptom_prediction": tabnet_disease,
        "final_diagnosis": disease_diagnosis
    }

    # --- Fuse Insect Predictions ---
    is_yolo_insect_present = "Present" in yolo_insect
    is_tabnet_insect_present = "Present" in tabnet_insect

    if is_yolo_insect_present and is_tabnet_insect_present:
        insect_diagnosis = "Confirmed: Insect Present (High Confidence)"
    elif is_yolo_insect_present:
        insect_diagnosis = "Possible: Insect Present (Image Only)"
    elif is_tabnet_insect_present:
        insect_diagnosis = "Possible: Insect Present (Symptoms Reported)"
    else:
        insect_diagnosis = "No Insect Detected"

    final_output["insect_analysis"] = {
        "image_detection": yolo_insect.strip(),
        "symptom_prediction": tabnet_insect,
        "final_diagnosis": insect_diagnosis
    }

    # The function now returns the dictionary instead of printing it
    return final_output