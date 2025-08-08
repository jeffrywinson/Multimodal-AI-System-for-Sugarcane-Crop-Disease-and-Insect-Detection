# fusion.py (Combined NN Power with Web-Friendly Interface)
import torch
import torch.nn as nn
import json

# --- Your powerful neural network architecture ---
# This part is copied directly from your main branch version.
BEST_PARAMS = {
    'n_layers': 3,
    'n_units_l0': 45,
    'n_units_l1': 60,
    'n_units_l2': 55,
    'optimizer': 'Adam',
    'lr': 0.00626330958689155
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

# --- Your adapted prediction logic ---
# This function now returns the result instead of printing it.
def run_nn_fusion(model_inputs, yolo_disease_str, yolo_insect_str):
    """
    Runs the neural network prediction.
    This is an internal function that performs the core ML inference.
    """
    model = FusionNet()
    # Ensure the model file 'fusion_model.pth' is available to the web server
    model.load_state_dict(torch.load('fusion_model.pth'))
    model.eval()

    input_tensor = torch.tensor(model_inputs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        probabilities = model(input_tensor)
    
    disease_prob = probabilities[0][0].item()
    insect_prob = probabilities[0][1].item()

    disease_present = "Confirmed: Disease Present (High Confidence)" if disease_prob > 0.75 else "Possible: Disease Present" if disease_prob > 0.5 else "No Disease Detected"
    insect_present = "Confirmed: Insect Present (High Confidence)" if insect_prob > 0.75 else "Possible: Insect Present" if insect_prob > 0.5 else "No Insect Detected"
    
    final_output = {
        "disease_analysis": {
            "image_detection": yolo_disease_str,
            "symptom_prediction_prob": f"{model_inputs[2]:.4f}",
            "fused_probability": f"{disease_prob:.4f}",
            "final_diagnosis": disease_present
        },
        "insect_analysis": {
            "image_detection": yolo_insect_str,
            "symptom_prediction_prob": f"{model_inputs[3]:.4f}",
            "fused_probability": f"{insect_prob:.4f}",
            "final_diagnosis": insect_present
        }
    }
    # --- KEY CHANGE ---
    # We return the dictionary so the web app can use it.
    return final_output

# --- The Web App's Main Function ---
# This is the function the website will call. It's your friend's function,
# but now it prepares data for and calls your neural network.
def fuse_predictions(image_name, yolo_disease, tabnet_disease_prob, yolo_insect, tabnet_insect_prob):
    """
    Main entry point for the web application. It takes raw inputs,
    prepares them for the neural network, and returns the final JSON.
    """
    # Prepare the list of floats required by the neural network
    nn_inputs = [
        1.0 if "Present" in yolo_disease else 0.0,
        1.0 if "Present" in yolo_insect else 0.0,
        tabnet_disease_prob,
        tabnet_insect_prob
    ]
    
    # Run the neural network fusion
    final_result = run_nn_fusion(nn_inputs, yolo_disease, yolo_insect)
    
    # Add the image_name to the final output
    final_result['image_name'] = image_name
    
    return final_result