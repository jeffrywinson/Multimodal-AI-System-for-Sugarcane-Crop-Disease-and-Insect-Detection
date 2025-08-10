# prepare_round2_tabnet.py (Definitive Final Version with Selective Training)

import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import argparse

# ==============================================================================
# 1. DEFINE DOMAIN-SPECIFIC RULES FOR ROUND 2
# ==============================================================================

# For Insect Detection (Early Shoot Borer / Internode Borer / No Insect)
insect_rules = [
    ("Is the crop ≤ 120 days old (i.e., within first 4 months)?", "ESB_positive"),
    ("Did the damage start after 4 months from planting?", "INB_positive"),
    ("Did you first notice the damage between March and June?", "ESB_positive"),
    ("Did you first notice the damage between June and December?", "INB_positive"),
    ("Did attacks start very soon after planting (within about 15 days)?", "ESB_positive"),
    ("Did the damage stop appearing after 4 months from planting?", "ESB_positive"),
    ("Is the peak damage appearing around 7–9 months after planting?", "INB_positive"),
    ("Does the damage seem to start from the lower part of the stalk?", "ESB_positive"),
    ("Does the damage seem to start from the upper part of the stalk?", "INB_positive"),
    ("Are bore holes within 15 cm from the soil?", "ESB_positive"),
    ("Are bore holes on the upper internodes?", "INB_positive"),
    ("Are some stalk internodes malformed or constricted?", "INB_positive"),
    ("Do you see small aerial roots appearing on the stalks above the ground?", "INB_positive"),
    ("When you pull the dead central shoot, does it come out easily?", "ESB_positive"),
    ("Does the pulled shoot have a foul or bad smell?", "ESB_positive"),
    ("Is the pulled shoot difficult to remove?", "INB_positive"),
    ("When removed, is there no bad smell from the shoot?", "INB_positive"),
    ("Are insect eggs present on the underside of the lower leaves?", "ESB_positive"),
    ("Are insect eggs present on the underside of the top leaves?", "INB_positive"),
    ("Are the eggs flat, white, and smaller than 1 mm?", "ESB_positive"),
    ("Does the larva have 5 visible stripes along its body?", "ESB_positive"),
    ("Does the larva have only 4 visible stripes along its body?", "INB_positive"),
    ("Is fresh powder-like excreta seen near the base of the stalk?", "ESB_positive"),
    ("Is fresh excreta seen on the upper internodes?", "INB_positive"),
    ("In the early stages, do you see only green leaf scraping without bore holes?", "ESB_positive"),
    ("Has the damage occurred only after internodes have fully developed?", "INB_positive"),
    ("Along with dead shoots, do you see bunchy or abnormal top growth?", "INB_positive"),
    ("Did you apply a high dose of nitrogen/urea before the damage started?", "general_borer_positive"),
    ("Was trash mulching done early in the crop stage?", "ESB_management_positive"),
    ("Was earthing-up done to cover the lower stalk area?", "ESB_management_positive")
]

# For Dead Heart Detection
dead_heart_rules = [
    ("Have you seen the central growing point of the stalk damaged or dead?", "dead_heart_confirmation_positive"),
    ("Is the dead central shoot straw-coloured?", "dead_heart_confirmation_positive"),
    ("After pulling the dead-heart, do you find maggots that look like secondary feeders (not primary borers)?", "dead_heart_confirmation_positive"),
    ("After pulling the dead-heart, do you find fresh live larvae inside the affected stem?", "dead_heart_confirmation_positive"),
    ("Is most of the visible damage inside the stem rather than outside?", "dead_heart_confirmation_positive"),
    ("Have you noticed insect attack when the leaves are still developing and soft?", "dead_heart_confirmation_positive"),
    ("Was the crop planted within the last 15 days?", "dead_heart_timing_negative"),
    ("Have you never seen moths flying during daytime?", "dead_heart_cause_neutral"),
    ("Have you observed mating or egg-laying activity mostly at night?", "dead_heart_cause_neutral"),
    ("Were any biological control insects released in the field?", "dead_heart_cause_neutral"),
    ("Have you seen fully grown moths that are straw to light brown in colour?", "dead_heart_cause_positive"),
    ("Is the central shoot of young plants dry, brown, or straw-colored?", "dead_heart_confirmation_positive"),
    ("Does the central shoot come out easily when pulled gently?", "dead_heart_confirmation_positive"),
    ("Does the pulled shoot emit a foul or rotten odor?", "dead_heart_confirmation_positive"),
    ("Are leaves around the central shoot yellowing, wilting, or drying?", "dead_heart_confirmation_positive"),
    ("Do patches of plants show multiple stalks with dried or dead centers?", "dead_heart_confirmation_positive"),
    ("Has the percentage of dead hearts increased after recent rains or waterlogging?", "dead_heart_confirmation_positive"),
    ("Are affected plants stunted or shorter than surrounding healthy plants?", "dead_heart_confirmation_positive"),
    ("Do affected plants fail to produce new green shoots or leaves?", "dead_heart_confirmation_positive"),
    ("Are there soft, hollow, or tunnel-like areas inside the affected stalks?", "dead_heart_confirmation_positive"),
    ("Have you seen bunchy or abnormal growth at the top of affected stalks?", "dead_heart_confirmation_positive"),
    ("Are soil moisture and drainage poor in areas where dead hearts appear?", "dead_heart_confirmation_positive"),
    ("Are there no dry central shoots in the field?", "dead_heart_confirmation_negative"),
    ("Is plant height uniform and normal throughout the field?", "dead_heart_confirmation_negative"),
    ("When pulling the central shoot, is it firmly attached without coming out easily?", "dead_heart_confirmation_negative"),
    ("Does the shoot base smell fresh with no rotting or foul odor?", "dead_heart_confirmation_negative"),
    ("Are leaves healthy, green, and not wilting near the central shoot?", "dead_heart_confirmation_negative"),
    ("Do you have no patches with multiple dead or dried shoots?", "dead_heart_confirmation_negative"),
    ("Have symptoms decreased after improved irrigation or fertilization?", "dead_heart_confirmation_negative"),
    ("Is there no recurrence of dead heart symptoms from previous seasons?", "dead_heart_confirmation_negative")
]


# ==============================================================================
# 2. ADVANCED DATA GENERATION (ANTI-OVERFITTING VERSION)
# ==============================================================================

def generate_multi_class_insect_data(filename, rules, num_rows=20000):
    """Generates multi-class data with fuzzy scoring to reduce overfitting."""
    print(f"Generating fuzzy multi-class data for {filename}...")
    questions = [rule[0] for rule in rules]
    score_types = [rule[1] for rule in rules]
    column_headers = [f'q_{i+1}' for i in range(len(questions))]
    data = []
    for _ in range(num_rows):
        answers = [random.choice(['Yes', 'No']) for _ in questions]
        esb_score = 0
        inb_score = 0
        for i, answer in enumerate(answers):
            if answer == 'No': continue
            rule_type = score_types[i]
            if rule_type == "ESB_positive": esb_score += 2 + random.uniform(-0.5, 0.5)
            elif rule_type == "INB_positive": inb_score += 2 + random.uniform(-0.5, 0.5)
            elif rule_type == "general_borer_positive":
                esb_score += 1 + random.uniform(-0.5, 0.5)
                inb_score += 1 + random.uniform(-0.5, 0.5)
            elif rule_type == "ESB_management_positive": esb_score -= 3
        threshold = 5 + random.uniform(-1, 1)
        if esb_score > threshold and esb_score > inb_score: final_class = "Early Shoot Borer"
        elif inb_score > threshold and inb_score > esb_score: final_class = "Internode Borer"
        else: final_class = "No Insect"
        row = answers + [final_class]
        data.append(row)
    df = pd.DataFrame(data, columns=column_headers + ['Presence'])
    df.to_csv(filename, index=False)
    print(f"Successfully created {filename}.")

def generate_binary_dead_heart_data(filename, rules, num_rows=20000):
    """Generates binary data with fuzzy scoring (the balanced approach)."""
    print(f"Generating balanced fuzzy binary data for {filename}...")
    questions = [rule[0] for rule in rules]
    score_types = [rule[1] for rule in rules]
    column_headers = [f'q_{i+1}' for i in range(len(questions))]
    data = []
    for _ in range(num_rows):
        answers = [random.choice(['Yes', 'No']) for _ in questions]
        score = 0
        for i, answer in enumerate(answers):
            if answer == 'No': continue
            rule_type = score_types[i]
            # We keep the fuzzy scoring, which is good
            if "confirmation_positive" in rule_type: score += 3 + random.uniform(-0.5, 0.5)
            elif "cause_positive" in rule_type: score += 1 + random.uniform(-0.5, 0.5)
            elif "negative" in rule_type: score -= 2
        
        # We REMOVED the aggressive "label noise" part from the last version
        final_class = "Present" if score > (6 + random.uniform(-1, 1)) else "Not Present"
        
        row = answers + [final_class]
        data.append(row)
        
    df = pd.DataFrame(data, columns=column_headers + ['Presence'])
    df.to_csv(filename, index=False)
    print(f"Successfully created {filename}.")

# ==============================================================================
# 3. TABNET TRAINING SCRIPT (WITH REGULARIZATION)
# ==============================================================================

def train_tabnet_model(csv_path, model_name, best_params):
    """Loads data, trains a TabNet model with the best parameters, and saves it."""
    print(f"\n--- Starting definitive training for {model_name} using best parameters ---")
    
    df = pd.read_csv(csv_path)
    target = 'Presence'
    
    X = df.drop(columns=[target])
    y = df[target]

    categorical_features = list(X.columns)
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    for col in categorical_features:
        le_x = LabelEncoder()
        X[col] = le_x.fit_transform(X[col])

    X_train, X_val, y_train, y_val = train_test_split(X.values, y_encoded, test_size=0.2, random_state=42)

    scheduler_params = dict(mode="max", patience=10, factor=0.2, verbose=True)

    clf = TabNetClassifier(
        n_d=best_params['n_da'],
        n_a=best_params['n_da'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        lambda_sparse=best_params['lambda_sparse'],
        mask_type=best_params['mask_type'],
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 1e-5)),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=scheduler_params,
        seed=42,
        verbose=1
    )

    # Train for a long time to ensure the model fully converges
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['validation'],
        patience=40,
        max_epochs=300,
        batch_size=2048
    )

    saved_model_path = clf.save_model(f'./{model_name}_model')
    print(f"--- Definitive model saved at {saved_model_path} ---")

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("--- Preparing Final, Optimized TabNet Models for Round 2 ---")
    
    # Part 1: Generate Data
    generate_multi_class_insect_data('sugarcane_insect_data_r2.csv', insect_rules)
    generate_binary_dead_heart_data('sugarcane_deadheart_data_r2.csv', dead_heart_rules)
    
    # Part 2: Train Final Models with the Best Parameters You Found
    
    # These are your winning parameters from the Optuna run
    best_insect_params = {
        'mask_type': 'sparsemax',
        'n_da': 40,
        'n_steps': 6,
        'gamma': 1.5308803108381908,
        'lambda_sparse': 0.0007525346361025023,
        'weight_decay': 0.00023110713728597548,
        'lr': 0.025707344165775824
    }
    
    best_disease_params = {
        'mask_type': 'entmax',
        'n_da': 24,
        'n_steps': 7,
        'gamma': 1.97595869478892,
        'lambda_sparse': 0.00011335382574385305,
        'weight_decay': 0.00016001843059440895,
        'lr': 0.022143722355915087
    }

    # Train both models using their respective best parameters
    train_tabnet_model(
        csv_path='sugarcane_insect_data_r2.csv', 
        model_name='tabnet_insect',
        best_params=best_insect_params
    )
    
    train_tabnet_model(
        csv_path='sugarcane_deadheart_data_r2.csv', 
        model_name='tabnet_disease',
        best_params=best_disease_params
    )
    
    print("\n--- All FINAL TabNet models for Round 2 have been prepared successfully! ---")