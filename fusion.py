# fusion.py (Final Version with Summary)
import argparse
import json

def fuse_predictions(image_name, yolo_disease, tabnet_disease, yolo_insect, tabnet_insect):
    final_output = {
        "image_name": image_name,
        "disease_analysis": {},
        "insect_analysis": {}
    }

    # --- Fuse Disease Predictions ---
    # The logic here is slightly adjusted to make the final diagnosis simpler for parsing.
    is_yolo_disease_present = yolo_disease == 'Present'
    is_tabnet_disease_present = tabnet_disease == 'Present'

    if is_yolo_disease_present and is_tabnet_disease_present:
        disease_diagnosis = "Confirmed: Disease Present (High Confidence)"
    elif is_yolo_disease_present:
        disease_diagnosis = "Possible: Disease Present (Image Only)"
    elif is_tabnet_disease_present:
        disease_diagnosis = "Possible: Disease Present (Symptoms Reported)"
    else:
        disease_diagnosis = "No Disease Detected"
        
    final_output["disease_analysis"] = {
        "image_detection": yolo_disease,
        "symptom_prediction": tabnet_disease,
        "final_diagnosis": disease_diagnosis
    }

    # --- Fuse Insect Predictions ---
    is_yolo_insect_present = yolo_insect == 'Present'
    is_tabnet_insect_present = tabnet_insect == 'Present'

    if is_yolo_insect_present and is_tabnet_insect_present:
        insect_diagnosis = "Confirmed: Insect Present (High Confidence)"
    elif is_yolo_insect_present:
        insect_diagnosis = "Possible: Insect Present (Image Only)"
    elif is_tabnet_insect_present:
        insect_diagnosis = "Possible: Insect Present (Symptoms Reported)"
    else:
        insect_diagnosis = "No Insect Detected"

    final_output["insect_analysis"] = {
        "image_detection": yolo_insect,
        "symptom_prediction": tabnet_insect,
        "final_diagnosis": insect_diagnosis
    }

    # --- Print the Detailed JSON Output (as before) ---
    print(json.dumps(final_output, indent=4))


    # --- NEW: Print the Final Simplified Summary ---
    # This section parses the final diagnosis strings we just created.
    print("\n-------------------------------------")
    print("Final Output:")

    if "No Disease Detected" in disease_diagnosis:
        print("Crop disease not present")
    else:
        print("Crop disease present")

    if "No Insect Detected" in insect_diagnosis:
        print("Crop insect not present")
    else:
        print("Crop insect present")
    print("-------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse predictions from all models.")
    parser.add_argument("--image_name", type=str, required=True)
    parser.add_argument("--yolo_disease", type=str, required=True)
    parser.add_argument("--tabnet_disease", type=str, required=True)
    parser.add_argument("--yolo_insect", type=str, required=True)
    parser.add_argument("--tabnet_insect", type=str, required=True)
    
    args = parser.parse_args()
    fuse_predictions(
        args.image_name,
        args.yolo_disease,
        args.tabnet_disease,
        args.yolo_insect,
        args.tabnet_insect
    )