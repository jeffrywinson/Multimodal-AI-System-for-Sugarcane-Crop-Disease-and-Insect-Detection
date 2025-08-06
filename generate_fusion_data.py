# generate_fusion_data.py (Corrected for Area Simulation)
import pandas as pd
import random
import numpy as np # Import numpy for more control

data = []
for _ in range(20000):
    # --- MODIFIED LINE ---
    # Simulate the YOLO disease output as an area percentage (a float from 0.0 to 1.0)
    # We'll make smaller areas more common.
    yolo_disease_area = max(0, np.random.normal(loc=0.05, scale=0.1)) # Normal distribution centered at 0.05
    if random.random() > 0.7: # 70% chance of having no disease area
        yolo_disease_area = 0.0
    # --- END OF MODIFICATION ---
    
    # Keep the insect count as an integer
    yolo_insect_count = random.choice([0, 0, 0, 1, 1, 2])
    
    # Simulate TabNet outputs
    tabnet_disease_proba = random.random()
    tabnet_insect_proba = random.random()

    # Determine the ground truth based on the simulated inputs
    # Now the rule uses the area percentage
    disease_present = 1 if (yolo_disease_area > 0.01 or tabnet_disease_proba > 0.6) else 0
    insect_present = 1 if (yolo_insect_count > 0 or tabnet_insect_proba > 0.6) else 0
    
    # Add some noise
    if random.random() < 0.1: disease_present = 1 - disease_present
    if random.random() < 0.1: insect_present = 1 - insect_present

    data.append([yolo_disease_area, yolo_insect_count, tabnet_disease_proba, tabnet_insect_proba, disease_present, insect_present])

columns = ['yolo_disease', 'yolo_insect', 'tabnet_disease', 'tabnet_insect', 'is_disease', 'is_insect']
df = pd.DataFrame(data, columns=columns)
df.to_csv('fusion_training_data.csv', index=False)
print("Generated fusion_training_data.csv successfully with simulated area percentage.")