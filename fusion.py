import math

def get_fused_prediction_rules(yolo_disease_area, yolo_insect_count, tabnet_disease_prob, tabnet_insect_class):
    """
    UPDATED VERSION:
    1. Fixes mismatched key names for fused probability.
    2. Calculates a certainty score for the insect diagnosis.
    3. Makes a primary decision and returns ONLY the most relevant analysis.
    """
    
    # --- 1. Disease Score Calculation ---
    normalized_yolo_score = min(yolo_disease_area / 0.2, 1.0)
    disease_score = (normalized_yolo_score * 0.6) + (tabnet_disease_prob * 0.4)
    
    if disease_score > 0.5:
        disease_diagnosis = "Dead Heart Present" # More specific diagnosis
    else:
        disease_diagnosis = "Healthy"
        
    disease_analysis = {
        "yolo_output": f"{yolo_disease_area:.4f} area",
        "tabnet_probability": f"{tabnet_disease_prob:.4f}",
        "fused_probability": f"{disease_score:.4f}",  # FIX: Renamed key to match JS
        "final_diagnosis": disease_diagnosis
    }
        
    # --- 2. Insect Diagnosis Calculation ---
    decision_rule = ""
    insect_probability = 0.0 # FIX: Added a probability score for insects

    if yolo_insect_count > 0:
        if tabnet_insect_class != "No Insect":
            insect_diagnosis = tabnet_insect_class
            decision_rule = "Visual evidence confirmed by symptoms."
            insect_probability = 0.95 # High confidence
        else:
            insect_diagnosis = "Insect Detected (Type Unconfirmed)"
            decision_rule = "Visual evidence overrides conflicting symptom data."
            insect_probability = 0.85 # Still high confidence due to visual
            
    else: # yolo_insect_count == 0
        if tabnet_insect_class != "No Insect":
            insect_diagnosis = tabnet_insect_class
            decision_rule = "Symptom-based detection without visual confirmation."
            insect_probability = 0.70 # Moderate confidence
        else:
            insect_diagnosis = "Healthy"
            decision_rule = "No visual or symptom-based evidence found."
            insect_probability = 1.0 - tabnet_disease_prob # Confidence in being healthy

    insect_analysis = {
        "yolo_output": f"{yolo_insect_count} detections",
        "tabnet_classification": tabnet_insect_class,
        "decision_logic": decision_rule,
        "fused_probability": f"{insect_probability:.4f}", # FIX: Added the new key
        "final_diagnosis": insect_diagnosis
    }

    # --- 3. Primary Decision Logic ---
    # Decide whether to show the disease OR the insect analysis.
    # We prioritize showing the result with the higher visual evidence.
    
    final_output = {}
    if yolo_disease_area > 0.01 or yolo_insect_count > 0:
        # If there's strong visual evidence for either, pick the stronger one
        if (yolo_disease_area * 3) > yolo_insect_count: # Weight disease area more
             final_output['analysis_type'] = 'disease'
             final_output['result'] = disease_analysis
        else:
            final_output['analysis_type'] = 'insect'
            final_output['result'] = insect_analysis
    else:
        # If no visual evidence, rely on the TabNet scores
        if tabnet_disease_prob > 0.5:
            final_output['analysis_type'] = 'disease'
            final_output['result'] = disease_analysis
        elif tabnet_insect_class != "No Insect":
             final_output['analysis_type'] = 'insect'
             final_output['result'] = insect_analysis
        else:
            # If still nothing, default to showing the disease analysis as "Healthy"
            final_output['analysis_type'] = 'disease'
            final_output['result'] = disease_analysis

    return final_output