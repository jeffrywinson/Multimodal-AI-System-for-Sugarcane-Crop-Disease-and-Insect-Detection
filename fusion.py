# fusion.py
import argparse
import json

def fuse_predictions(image_name, yolo_disease, tabnet_disease, yolo_insect, tabnet_insect):
    final_output = {
        "image_name": image_name,
        "disease_analysis": {},
        "insect_analysis": {}
    }

    # --- Fuse Disease Predictions ---
    is_yolo_disease = yolo_disease != 'None'
    is_tabnet_disease = tabnet_disease == 'Present'

    if is_yolo_disease and is_tabnet_disease:
        final_diagnosis = f"Confirmed: {yolo_disease} (High Confidence)"
    elif is_yolo_disease:
        final_diagnosis = f"Possible: {yolo_disease} (Image Only)"
    elif is_tabnet_disease:
        final_diagnosis = "Possible Disease (Symptoms Reported)"
    else:
        final_diagnosis = "No Disease Detected"

    final_output["disease_analysis"] = {
        "image_detection": yolo_disease,
        "symptom_prediction": tabnet_disease,
        "final_diagnosis": final_diagnosis
    }

    # --- Fuse Insect Predictions ---
    is_yolo_insect = yolo_insect != 'None'
    is_tabnet_insect = tabnet_insect == 'Present'

    if is_yolo_insect and is_tabnet_insect:
        final_diagnosis = f"Confirmed: {yolo_insect} (High Confidence)"
    elif is_yolo_insect:
        final_diagnosis = f"Possible: {yolo_insect} (Image Only)"
    elif is_tabnet_insect:
        final_diagnosis = "Possible Insect (Symptoms Reported)"
    else:
        final_diagnosis = "No Insect Detected"

    final_output["insect_analysis"] = {
        "image_detection": yolo_insect,
        "symptom_prediction": tabnet_insect,
        "final_diagnosis": final_diagnosis
    }

    print(json.dumps(final_output, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse predictions from all models.")
    parser.add_argument("--image_name", type=str, required=True)
    parser.add_argument("--yolo_disease", type=str, required=True)
    parser.add_argument("--tabnet_disease", type=str, required=True)
    parser.add_argument("--yolo_insect", type=str, required=True)
    parser.add_argument("--tabnet_insect", type=str, required=True)

    args = parser.parse_args()
    fuse_predictions(args.image_name, args.yolo_disease, args.tabnet_disease, args.yolo_insect, args.tabnet_insect)