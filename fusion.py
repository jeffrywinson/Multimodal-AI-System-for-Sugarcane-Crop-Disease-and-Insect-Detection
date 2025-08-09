# fusion.py (Definitive Final Version)
import torch
import torch.nn as nn
import argparse
import json

# --- This dictionary and class MUST match your final Round 2 training script ---
# (Using the best parameters from your Optuna run)
BEST_PARAMS = {
    'n_layers': 4,
    'n_units_l0': 61,
    'n_units_l1': 26,
    'n_units_l2': 30,
    'n_units_l3': 17,
    'optimizer': 'Adam',
    'lr': 0.004255325260776246
}

class FusionNetR2(nn.Module):
    def __init__(self):
        super(FusionNetR2, self).__init__()
        layers = []
        in_features = 6
        for i in range(BEST_PARAMS['n_layers']):
            out_features = BEST_PARAMS[f'n_units_l{i}']
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, 2))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# --------------------------------------------------------------------

def get_fused_prediction(yolo_disease_area, yolo_insect_count, tabnet_disease_prob, tabnet_insect_class):
    """
    Takes all model outputs, runs the fusion network, and returns the final JSON.
    """
    model = FusionNetR2()
    
    # --- THIS IS THE FIX ---
    # Load the correct Round 2 model file
    model.load_state_dict(torch.load('fusion_model_r2.pth'))
    # -----------------------
    
    model.eval()

    is_esb = 1 if tabnet_insect_class == "Early Shoot Borer" else 0
    is_inb = 1 if tabnet_insect_class == "Internode Borer" else 0
    is_no_insect = 1 if tabnet_insect_class == "No Insect" else 0
    
    model_inputs = [yolo_disease_area, yolo_insect_count, tabnet_disease_prob, is_esb, is_inb, is_no_insect]
    input_tensor = torch.tensor(model_inputs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        probabilities = model(input_tensor)
    
    disease_prob = probabilities[0][0].item()
    insect_prob = probabilities[0][1].item()

    disease_diagnosis = "Present" if disease_prob > 0.5 else "Not Present"
    insect_diagnosis = tabnet_insect_class if insect_prob > 0.5 and tabnet_insect_class != "No Insect" else "Not Present"
    
    final_output = {
        "dead_heart_analysis": {
            "yolo_output": f"{yolo_disease_area:.4f} area",
            "tabnet_probability": f"{tabnet_disease_prob:.4f}",
            "fused_probability": f"{disease_prob:.4f}",
            "final_diagnosis": disease_diagnosis
        },
        "insect_analysis": {
            "yolo_output": f"{yolo_insect_count} detections",
            "tabnet_classification": tabnet_insect_class,
            "fused_probability": f"{insect_prob:.4f}",
            "final_diagnosis": insect_diagnosis
        }
    }
    return final_output
# This block allows the script to still be run from the command line for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse Round 2 model outputs.")
    parser.add_argument("--yolo_disease", type=float, required=True)
    parser.add_argument("--yolo_insect", type=float, required=True)
    parser.add_argument("--tabnet_disease", type=float, required=True)
    parser.add_argument("--tabnet_insect", type=str, required=True)
    
    args = parser.parse_args()
    
    # Call the main function and print its return value
    final_json = get_fused_prediction(args.yolo_disease, args.yolo_insect, args.tabnet_disease, args.tabnet_insect)
    print(json.dumps(final_json, indent=4))