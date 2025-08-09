# prepare_tabnet_models.py (Final Complete Version)

import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import argparse # Import argparse to handle command-line flags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DEFINE DOMAIN-SPECIFIC RULES (Sugarcane for Round 2)
# THIS IS THE SECTION THAT WAS MISSING IN YOUR SNIPPET
# ==============================================================================

# For Insect Detection (Early Shoot Borer) - 30 questions
insect_rules = [
    ("Have you seen pale greyish-brown moths with black dots on the wings in your field?", "positive"),
    ("Do you see thin white or yellow flat egg clusters under leaf sheaths?", "positive"),
    ("Have you found white-striped caterpillars when splitting stems?", "positive"),
    ("Do you see small round bore holes just above soil level on the cane?", "positive"),
    ("Have you noticed fine powder-like waste (frass) near the bore holes?", "positive"),
    ("When cutting the stem near the base, do you find internal tunnels?", "positive"),
    ("Have you observed cane tops or whorls showing slight drooping before drying?", "positive"),
    ("Are there more insect sightings during evening or night time?", "positive"),
    ("Is pest incidence higher after warm, humid weather spells?", "positive"),
    ("Does pest damage often start from edges of the field?", "positive"),
    ("Have you delayed destruction of crop residue after harvest?", "positive"),
    ("Was sugarcane planted late (after February) this season?", "positive"),
    ("Have you skipped using recommended biological control like Trichogramma?", "positive"),
    ("Has there been no intercrop with daincha/legumes this year?", "positive"),
    ("Have you seen larvae making larger oval exit holes before pupation?", "positive"),
    ("Do infested canes often have multiple bore entry points?", "positive"),
    ("Are stubble and trash left in the field after harvest?", "positive"),
    ("Have you observed damaged stems attracting ants or other insects?", "positive"),
    ("Does infestation flare up after irrigation during hot months?", "positive"),
    ("Is there an increase in moth activity after rains?", "positive"),
    ("Have no moths been sighted during evening field walk?", "negative"),
    ("Is there no presence of egg clusters under leaf sheaths?", "negative"),
    ("Are there no visible bore holes at the cane base?", "negative"),
    ("When splitting stems, are no larvae or tunnels found?", "negative"),
    ("Have you destroyed all crop residues before new planting?", "negative"),
    ("Did you plant in December–January this year?", "negative"),
    ("Did you intercrop with pest-reducing crops like daincha or green gram?", "negative"),
    ("Are biological control measures applied regularly?", "negative"),
    ("Is moth activity low even in warm/humid weather?", "negative"),
    ("Have you maintained clean and weed-free field borders?", "negative")
]

# For Disease/Symptom Detection (Dead Heart) - 30 questions
dead_heart_rules = [
    ("Is the central shoot of the cane plant dry and brown in colour?", "positive"),
    ("Does the central shoot come out when pulled gently?", "positive"),
    ("Does the pulled shoot base smell sour, bad, or rotten?", "positive"),
    ("Are the leaves around the central shoot pale, yellow, or wilting?", "positive"),
    ("Do patches of the field show plants with dried centers?", "positive"),
    ("Are discoloured leaves appearing from the whorl (center of plant)?", "positive"),
    ("Is the sugarcane height uneven with some plants staying much shorter?", "positive"),
    ("Do affected plants feel light and brittle compared to healthy ones?", "positive"),
    ("Have dead centers appeared after heavy rains or waterlogging?", "positive"),
    ("Do affected plants in wet areas of the field rot faster?", "positive"),
    ("Is the soil around affected plants giving a foul odor?", "positive"),
    ("Have dead hearts increased after long dry spells followed by rain?", "positive"),
    ("Do you see more dead plants in poorly drained spots?", "positive"),
    ("Have tops of affected canes failed to produce new green leaves?", "positive"),
    ("Do damaged plants occur in low-lying areas more often?", "positive"),
    ("Are multiple plants in a row showing dried centers?", "positive"),
    ("Have you noticed dead hearts spreading quickly between plants?", "positive"),
    ("Do affected plants remain stunted even after fertiliser application?", "positive"),
    ("Are there wilted leaves that stay attached and do not fall off?", "positive"),
    ("Has this symptom appeared in previous seasons as well?", "positive"),
    ("Are all cane plants green with no dry central shoots?", "negative"),
    ("Is cane height uniform across the field?", "negative"),
    ("When the central shoot is pulled, is it firmly attached?", "negative"),
    ("Does the base of the shoot smell fresh without any rotting smell?", "negative"),
    ("Are leaf whorls healthy and producing new green leaves?", "negative"),
    ("Do no patches in the field show groups of dry plants?", "negative"),
    ("Are field drainage and water flow even after rainfall?", "negative"),
    ("Have symptoms reduced after applying fertilisers?", "negative"),
    ("Is there no repeat of this symptom from past seasons?", "negative"),
    ("Have drying plants recovered after irrigation?", "negative")
]

# ==============================================================================
# 2. ADVANCED DATA GENERATION WITH DYNAMIC WEIGHTS
# ==============================================================================
def generate_training_csv(filename, rules, num_rows=20000):
    """Generates a high-quality synthetic CSV using dynamic, rule-based weights."""
    print(f"Generating advanced synthetic data for {filename}...")
    
    questions = [rule[0] for rule in rules]
    score_types = [rule[1] for rule in rules]
    column_headers = [f'q_{i+1}' for i in range(len(questions))]
    
    data = []
    for _ in range(num_rows):
        answers = [random.choice(['Yes', 'No']) for _ in questions]
        symptom_score = 0
        positive_symptom_count = 0
        for i, answer in enumerate(answers):
            if answer == 'No': continue
            if score_types[i] == "positive":
                positive_symptom_count += 1
                symptom_score += 3 if i < 7 else 1
            elif score_types[i] == "negative":
                symptom_score -= 4 if i < 25 else 2
        
        is_present_base = symptom_score > (positive_symptom_count * 0.5)
        presence = 'Present' if is_present_base else 'Not Present'
        if random.random() < 0.15:
            presence = 'Not Present' if presence == 'Present' else 'Present'

        row = answers + [presence]
        data.append(row)
        
    df = pd.DataFrame(data, columns=column_headers + ['Presence'])
    df.to_csv(filename, index=False)
    print(f"Successfully created {filename} with {len(questions)} columns and {num_rows} rows.")

# ==============================================================================
# 3. TABNET TRAINING FUNCTION (with Feature Importance)
# ==============================================================================
def train(csv_path, model_name, rules, epochs, patience, lr, generate_charts):
    """Trains a TabNet model and optionally visualizes feature importances."""
    print(f"\n--- Starting training for {model_name} ---")

    df = pd.read_csv(csv_path)
    target = 'Presence'
    questions = [rule[0] for rule in rules]
    
    categorical_features = [col for col in df.columns if col != target]
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    optimizer_params = dict(lr=lr, weight_decay=1e-5)
    clf = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=optimizer_params, verbose=1, seed=42)

    print(f"Training {model_name}...")
    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=['validation'],
        patience=patience,
        max_epochs=epochs
    )

    saved_model_path = clf.save_model(f'./{model_name}_model')
    print(f"--- Training finished. Model saved at {saved_model_path} ---")

    if generate_charts:
        print(f"\n--- Generating Feature Importance Chart for {model_name} ---")
        importances = clf.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'questions': [f"Q{i+1}" for i in range(len(questions))],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 12))
        plt.barh(feature_importance_df['questions'], feature_importance_df['importance'])
        plt.xlabel("Importance Score")
        plt.ylabel("Question Number")
        plt.title(f"TabNet Feature Importance for {model_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        chart_filename = f'{model_name}_feature_importance.png'
        plt.savefig(chart_filename)
        print(f"Feature importance chart saved as {chart_filename}")

# ==============================================================================
# 4. MAIN EXECUTION BLOCK (Now with a command-line parser)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and train TabNet models for Round 2.")
    parser.add_argument(
        '--generate_charts', 
        action='store_true',
        help="If set, generate and save feature importance charts (slower)."
    )
    args = parser.parse_args()
    
    print("--- Starting Round 2 TabNet Model Preparation ---")
    
    generate_training_csv('sugarcane_insect_data.csv', insect_rules)
    generate_training_csv('sugarcane_deadheart_data.csv', dead_heart_rules)
    
    OPTIMAL_EPOCHS = 150
    OPTIMAL_PATIENCE = 20
    OPTIMAL_LR = 0.01

    train(
        csv_path='sugarcane_insect_data.csv', 
        model_name='tabnet_insect',
        rules=insect_rules,
        epochs=OPTIMAL_EPOCHS, 
        patience=OPTIMAL_PATIENCE, 
        lr=OPTIMAL_LR,
        generate_charts=args.generate_charts
    )
    
    train(
        csv_path='sugarcane_deadheart_data.csv', 
        model_name='tabnet_disease',
        rules=dead_heart_rules,
        epochs=OPTIMAL_EPOCHS, 
        patience=OPTIMAL_PATIENCE, 
        lr=OPTIMAL_LR,
        generate_charts=args.generate_charts
    )
    
    print("\n--- All TabNet models have been prepared successfully! ---")