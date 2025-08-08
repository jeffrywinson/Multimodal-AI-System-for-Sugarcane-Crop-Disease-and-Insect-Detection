# fusion.py (Definitive Version - Matches Final Training Script)
import torch
import torch.nn as nn
import argparse
import json

# --- DEFINE THE WINNING ARCHITECTURE FOUND BY OPTUNA ---
# This dictionary MUST match the one in your final training script.
BEST_PARAMS = {
    'n_layers': 3,
    'n_units_l0': 45,
    'n_units_l1': 60,
    'n_units_l2': 55,
    'optimizer': 'Adam',
    'lr': 0.00626330958689155
}
# ---------------------------------------------------------------

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        layers = []
        in_features = 4
        # Dynamically build the network exactly as it was during training
        for i in range(BEST_PARAMS['n_layers']):
            out_features = BEST_PARAMS[f'n_units_l{i}']
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, 2))
        layers.append(nn.Sigmoid())
        
        # This self.network wrapper is the key to matching the saved file
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def run_fusion(inputs):
    # 1. Load the trained model into the correctly defined architecture
    model = FusionNet()
    model.load_state_dict(torch.load('fusion_model.pth'))
    model.eval()

    # 2. Prepare the input tensor
    input_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)

    # 3. Make a prediction
    with torch.no_grad():
        probabilities = model(input_tensor)
    
    disease_prob = probabilities[0][0].item()
    insect_prob = probabilities[0][1].item()

    # 4. Determine final diagnosis
    disease_present = "Present" if disease_prob > 0.5 else "Not Present"
    insect_present = "Present" if insect_prob > 0.5 else "Not Present"
    
    # 5. Create final JSON output
    final_output = {
        "disease_analysis": {
            "yolo_output": f"{inputs[0]} detections/area",
            "tabnet_probability": f"{inputs[2]:.4f}",
            "fused_probability": f"{disease_prob:.4f}",
            "final_diagnosis": disease_present
        },
        "insect_analysis": {
            "yolo_output": f"{inputs[1]} detections",
            "tabnet_probability": f"{inputs[3]:.4f}",
            "fused_probability": f"{insect_prob:.4f}",
            "final_diagnosis": insect_present
        }
    }
    print(json.dumps(final_output, indent=4))
    
    # And the simplified summary
    print("\n-------------------------------------")
    print("Final Output:")
    print(f"Crop disease {disease_present.lower()}")
    print(f"Crop insect {insect_present.lower()}")
    print("-------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse model outputs with a neural network.")
    parser.add_argument("--yolo_disease", type=float, required=True)
    parser.add_argument("--yolo_insect", type=float, required=True)
    parser.add_argument("--tabnet_disease", type=float, required=True)
    parser.add_argument("--tabnet_insect", type=float, required=True)
    
    args = parser.parse_args()
    model_inputs = [args.yolo_disease, args.yolo_insect, args.tabnet_disease, args.tabnet_insect]
    run_fusion(model_inputs)