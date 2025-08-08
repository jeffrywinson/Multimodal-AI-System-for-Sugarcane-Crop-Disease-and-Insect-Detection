# fusion.py
import torch
import torch.nn as nn
import json

BEST_PARAMS = {
    'n_layers': 3, 'n_units_l0': 45, 'n_units_l1': 60, 'n_units_l2': 55,
    'optimizer': 'Adam', 'lr': 0.00626330958689155
}

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        layers = []
        in_features = 4
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

def run_nn_fusion(model_inputs):
    model = FusionNet()
    model.load_state_dict(torch.load('fusion_model.pth'))
    model.eval()
    input_tensor = torch.tensor(model_inputs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(input_tensor)
    return probabilities[0][0].item(), probabilities[0][1].item()

def fuse_predictions(image_name, yolo_disease_str, tabnet_disease_prob, yolo_insect_str, tabnet_insect_prob):
    # Prepare the list of floats required by the neural network
    nn_inputs = [
        1.0 if "Present" in yolo_disease_str else 0.0,
        1.0 if "Present" in yolo_insect_str else 0.0,
        tabnet_disease_prob,
        tabnet_insect_prob
    ]
    
    fused_disease_prob, fused_insect_prob = run_nn_fusion(nn_inputs)
    
    # This creates the final output dictionary with corrected keys
    final_output = {
        "disease_analysis": {
            "image_detection": yolo_disease_str,  # <-- CHANGED from "yolo_output"
            "symptom_prediction": f"Probability Score: {tabnet_disease_prob:.4f}", # <-- CHANGED and improved
            "fused_probability": f"{fused_disease_prob:.4f}",
            "final_diagnosis": "Present" if fused_disease_prob > 0.5 else "Not Present"
        },
        "insect_analysis": {
            "image_detection": yolo_insect_str, # <-- CHANGED from "yolo_output"
            "symptom_prediction": f"Probability Score: {tabnet_insect_prob:.4f}", # <-- CHANGED and improved
            "fused_probability": f"{fused_insect_prob:.4f}",
            "final_diagnosis": "Present" if fused_insect_prob > 0.5 else "Not Present"
        }
    }
    final_output['image_name'] = image_name
    
    # Return the dictionary to be sent as JSON
    return final_output