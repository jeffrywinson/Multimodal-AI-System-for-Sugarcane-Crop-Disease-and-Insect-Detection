import math

def get_fused_prediction_rules(yolo_disease_area, yolo_insect_count, tabnet_disease_prob, tabnet_insect_class):
    """
    Takes all model outputs and uses a simple rule-based system (if/else)
    to determine the final diagnosis.
    """
    
    # --- 1. Rule-Based Disease Diagnosis ---
    
    # Normalize YOLO area to a 0-1 score to combine with TabNet's probability.
    # We'll cap the score at 1.0. A value of 0.2 area or more gets a max score.
    normalized_yolo_score = min(yolo_disease_area / 0.2, 1.0)
    
    # Create a weighted score. Give visual evidence from YOLO more weight.
    # Weighting: 60% YOLO, 40% TabNet
    disease_score = (normalized_yolo_score * 0.6) + (tabnet_disease_prob * 0.4)
    
    # Apply a simple threshold
    if disease_score > 0.5:
        disease_diagnosis = "Present"
    else:
        disease_diagnosis = "Not Present"
        
    # --- 2. Rule-Based Insect Diagnosis ---
    
    decision_rule = "" # To explain the logic in the output

    if yolo_insect_count > 0:
        # If we SEE an insect, we prioritize that visual evidence.
        if tabnet_insect_class != "No Insect":
            # Best case: Visuals and symptoms agree an insect is present.
            # We trust TabNet for the specific type.
            insect_diagnosis = tabnet_insect_class
            decision_rule = "Visual evidence confirmed by symptoms."
        else:
            # Conflict: We see an insect, but symptoms don't match a known type.
            # We must report the insect's presence but note the uncertainty.
            insect_diagnosis = "Insect Detected (Type Unconfirmed by Symptoms)"
            decision_rule = "Visual evidence overrides conflicting symptom data."
            
    else: # yolo_insect_count == 0
        # If we DON'T see an insect, we rely on the symptom data from TabNet.
        if tabnet_insect_class != "No Insect":
            # It's possible for borers to be present without being visible on the exterior.
            insect_diagnosis = tabnet_insect_class
            decision_rule = "Symptom-based detection without direct visual confirmation."
        else:
            # No visuals and no symptoms.
            insect_diagnosis = "Not Present"
            decision_rule = "No visual or symptom-based evidence found."

    # --- 3. Format Final Output ---
    # The structure is kept the same to ensure the frontend still works.
    
    final_output = {
        "dead_heart_analysis": {
            "yolo_output": f"{yolo_disease_area:.4f} area",
            "tabnet_probability": f"{tabnet_disease_prob:.4f}",
            "rule_based_score": f"{disease_score:.4f}", # Replaces 'fused_probability'
            "final_diagnosis": disease_diagnosis
        },
        "insect_analysis": {
            "yolo_output": f"{yolo_insect_count} detections",
            "tabnet_classification": tabnet_insect_class,
            "decision_logic": decision_rule, # Explains how the insect decision was made
            "final_diagnosis": insect_diagnosis
        }
    }
    return final_output