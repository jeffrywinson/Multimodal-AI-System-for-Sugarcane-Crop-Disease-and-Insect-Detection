# generate_fusion_data.py (More Noise to Prevent Overfitting)
import pandas as pd
import random
import numpy as np

print("Generating new, noisier training data for the Round 2 Fusion Network...")
data = []
insect_classes = ["Early Shoot Borer", "Internode Borer", "No Insect"]

# --- INCREASED DATASET SIZE FOR MORE ROBUSTNESS ---
for _ in range(30000): # Was 20000
    yolo_disease_area = max(0, np.random.normal(loc=0.05, scale=0.1)) if random.random() > 0.7 else 0.0
    yolo_insect_count = random.choice([0, 0, 0, 1, 2, 3])
    tabnet_disease_proba = random.random()
    tabnet_insect_class = random.choice(insect_classes)

    is_esb = 1 if tabnet_insect_class == "Early Shoot Borer" else 0
    is_inb = 1 if tabnet_insect_class == "Internode Borer" else 0
    is_no_insect = 1 if tabnet_insect_class == "No Insect" else 0
    
    disease_present = 1 if (yolo_disease_area > 0.01 or tabnet_disease_proba > 0.6) else 0
    insect_present = 1 if (yolo_insect_count > 0 and tabnet_insect_class != "No Insect") else 0
    
    # --- MORE NOISE: Increased from 0.15 to 0.25 ---
    if random.random() < 0.25: disease_present = 1 - disease_present
    if random.random() < 0.25: insect_present = 1 - insect_present
    
    row = [yolo_disease_area, yolo_insect_count, tabnet_disease_proba, is_esb, is_inb, is_no_insect, disease_present, insect_present]
    data.append(row)

columns = ['yolo_disease', 'yolo_insect', 'tabnet_disease', 'is_esb', 'is_inb', 'is_no_insect', 'is_disease_true', 'is_insect_true']
df = pd.DataFrame(data, columns=columns)
df.to_csv('fusion_training_data_r2.csv', index=False)
print("Successfully created a larger, noisier fusion_training_data_r2.csv.")