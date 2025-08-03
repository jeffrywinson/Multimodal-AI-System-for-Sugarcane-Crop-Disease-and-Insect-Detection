# fusion.py (Final Robust Version)
import argparse
import json

def fuse_predictions(image_name, yolo_disease, tabnet_disease, yolo_insect, tabnet_insect):
    final_output = {
        "image_name": image_name,
        "disease_analysis": {},
        "insect_analysis": {}
    }

    # --- Fuse Disease Predictions ---
    # CORRECTED LOGIC: Check if "Present" is IN the output string,
    # which ignores any extra text from the library's first run.
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
        "image_detection": yolo_disease.strip(), # Use .strip() to clean up whitespace
        "symptom_prediction": tabnet_disease,
        "final_diagnosis": disease_diagnosis
    }

    # --- Fuse Insect Predictions ---
    # Apply the same robust logic here for consistency
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
        "image_detection": yolo_insect.strip(), # Use .strip() to clean up whitespace
        "symptom_prediction": tabnet_insect,
        "final_diagnosis": insect_diagnosis
    }

    # --- Print the Detailed JSON Output ---
    print(json.dumps(final_output, indent=4))

    # --- Print the Final Simplified Summary ---
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