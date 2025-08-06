# fusion.py (Neural Network Version)
import torch
import torch.nn as nn
import argparse
import json

# Define the exact same neural network architecture as in the training script
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.layer_1 = nn.Linear(4, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

def run_fusion(inputs):
    # 1. Load the trained model
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
            "yolo_output": f"{inputs[0]} detections",
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
    # We now expect 4 numerical inputs
    parser.add_argument("--yolo_disease", type=float, required=True)
    parser.add_argument("--yolo_insect", type=float, required=True)
    parser.add_argument("--tabnet_disease", type=float, required=True)
    parser.add_argument("--tabnet_insect", type=float, required=True)
    
    args = parser.parse_args()
    # Create a list of the inputs
    model_inputs = [args.yolo_disease, args.yolo_insect, args.tabnet_disease, args.tabnet_insect]
    run_fusion(model_inputs)