# generate_data.py (Version 5 - Final, User-Verified Lists)
import pandas as pd
import random

# --- Disease Questions and Scoring Rules (Final List - 30 Questions) ---
disease_rules = [
    ("Is there a yellow halo around the spots?", "Positive"),
    ("Are the leaf spots circular with concentric rings?", "Positive"),
    ("Does the disease begin on the lower leaves?", "Positive"),
    ("Are the lesions expanding over time?", "Positive"),
    ("Is the center of the spot dry and brown?", "Positive"),
    ("Are multiple spots merging to form large blotches?", "Positive"),
    ("Does the leaf show signs of early yellowing?", "Positive"),
    ("Are stems or fruits also affected?", "Positive"),
    ("Are the affected leaves wilting?", "Positive"),
    ("Is the infection spreading upward on the plant?", "Positive"),
    ("Are concentric rings visible clearly on the leaves?", "Positive"),
    ("Is there any rotting seen on fruit?", "Positive"),
    ("Are the leaf margins turning brown?", "Positive"),
    ("Is the plant under moisture stress?", "Positive"),
    ("Is the disease more active during rainy days?", "Positive"),
    ("Are nearby tomato plants also showing similar symptoms?", "Positive"),
    ("Is there any black moldy growth on the lesion?", "Positive"),
    ("Does the disease affect the whole plant?", "Positive"),
    ("Is the spot size more than 5mm in diameter?", "Positive"),
    ("Are the lesions visible on both sides of the leaf?", "Positive"),
    ("Is the infection found only on mature leaves?", "Positive"),
    ("Are the leaf veins visible through the lesion?", "Positive"),
    ("Is the damage uniform across the field?", "Positive"),
    ("Was there previous history of Early Blight in this field?", "Positive"),
    ("Is the farmer using resistant tomato varieties?", "Negative"),
    ("Was any fungicide recently applied?", "Negative"),
    ("Was there poor air circulation in the field?", "Positive"),
    ("Was the field irrigated from overhead sprinklers?", "Positive"),
    ("Are pruning and sanitation practices followed?", "Negative"),
    ("Is there any other crop in the field showing similar spots?", "Positive")
]

# --- Insect Questions and Scoring Rules (Final List - 30 Questions) ---
insect_rules = [
    ("Is the pest in the image an armyworm?", "Positive"),
    ("Is the armyworm green in color?", "Positive"),
    ("Is the armyworm brown in color?", "Positive"),
    ("Is the armyworm found on the leaf top?", "Positive"),
    ("Is the armyworm found on the underside of the leaf?", "Positive"),
    ("Is the armyworm present on the stem?", "Positive"),
    ("Is the armyworm feeding on the crop?", "Positive"),
    ("Are visible bite marks present on the leaf?", "Positive"),
    ("Are there multiple armyworms in the image?", "Positive"),
    ("Is any frass (armyworm waste) visible near the pest?", "Positive"),
    ("Are eggs visible near the armyworm?", "Positive"),
    ("Are larvae of the armyworm visible?", "Positive"),
    ("Has the crop been attacked by armyworm in previous seasons?", "Positive"),
    ("Was pesticide recently applied to this crop area?", "Negative"),
    ("Is the armyworm population increasing?", "Positive"),
    ("Is the armyworm active during daylight hours?", "Positive"),
    ("Is the armyworm mostly active during night?", "Positive"),
    ("Is the leaf portion of the plant affected?", "Positive"),
    ("Is the stem portion of the plant affected?", "Positive"),
    ("Is the damage restricted to a small part of the crop?", "Negative"),
    ("Are nearby plants also showing signs of armyworm infestation?", "Positive"),
    ("Is the armyworm moving actively?", "Positive"),
    ("Are there signs of curled leaves due to feeding?", "Positive"),
    ("Has the armyworm damaged more than one section of the same plant?", "Positive"),
    ("Is there visible discoloration of the crop due to pest feeding?", "Positive"),
    ("Does the armyworm show striping or lines on its body?", "Positive"),
    ("Is the length of the armyworm greater than 20 mm?", "Positive"),
    ("Are any dead armyworms seen in the area (possibly due to pesticide)?", "Negative"),
    ("Is any chewing sound audible during the inspection?", "Positive"),
    ("Has any farmer nearby reported armyworm infestation in the last week?", "Positive")
]


def generate_training_csv(filename, rules, num_rows=10000):
    """
    Generates a synthetic CSV based on the final list of (question, score_type) rules.
    """
    print(f"Generating final synthetic data for {filename}...")
    
    questions = [rule[0] for rule in rules]
    score_types = [rule[1] for rule in rules]
    column_headers = [f'q_{i+1}' for i in range(len(questions))]
    
    data = []
    for _ in range(num_rows):
        answers = [random.choice(['Yes', 'No']) for _ in questions]
        
        # Weighted Logic based on explicit Positive/Negative score types
        symptom_score = 0
        positive_question_count = 0
        for i, answer in enumerate(answers):
            if score_types[i] == "Positive":
                positive_question_count += 1
                if answer == 'Yes':
                    symptom_score += 1
            elif score_types[i] == "Negative":
                if answer == 'Yes':
                    symptom_score -= 2 
        
        # The problem is present if the score of positive symptoms is high enough
        is_present_base = symptom_score > positive_question_count * 0.4

        # Add 15% random noise to make the data more realistic
        if random.random() < 0.15:
            presence = 'Not Present' if is_present_base else 'Present'
        else:
            presence = 'Present' if is_present_base else 'Not Present'

        row = answers + [presence]
        data.append(row)
        
    df = pd.DataFrame(data, columns=column_headers + ['Presence'])
    df.to_csv(filename, index=False)
    print(f"Successfully created {filename} with {len(questions)} question columns and {num_rows} rows.")

# --- Main script execution ---
print("--- Generating Final Training Files (Version 5) ---")

# Validate counts before generating
if len(disease_rules) == 30 and len(insect_rules) == 30:
    print(f"Validation successful: Found {len(disease_rules)} questions for disease and {len(insect_rules)} for insects.")
    
    generate_training_csv('crop_disease_training_data_final.csv', disease_rules)
    generate_training_csv('crop_insect_training_data_final.csv', insect_rules)

    print("\nProcess complete. You can now retrain your models on the definitive 'final' files.")
else:
    print("--- STOPPING ---")
    print(f"Error: Validation failed. Expected 30 questions for each category.")
    print(f"Found {len(disease_rules)} for disease and {len(insect_rules)} for insects.")
    print("Please correct the lists in the script and try again.")