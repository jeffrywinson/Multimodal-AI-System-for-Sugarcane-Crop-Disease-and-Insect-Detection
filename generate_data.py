# generate_data.py (Updated Version)
import pandas as pd
import random

# --- This function will now read questions from the user's original files ---
def get_questions_from_file(filepath, skiprows):
    try:
        df = pd.read_csv(filepath, skiprows=skiprows)
        # Handle the case of the extra empty column
        if len(df.columns) == 5:
            df.columns = ['Unnamed: 0', 'Sl.no', 'Question', 'Options', 'Unnamed: 4']
        else:
            df.columns = ['Unnamed: 0', 'Sl.no', 'Question', 'Options']

        return df['Question'].dropna().tolist()
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath}. Cannot generate data based on it.")
        return []

def generate_training_csv(filename, questions, num_rows=10000): # Increased num_rows
    """
    Generates a larger, more realistic synthetic CSV file for training.
    """
    if not questions:
        print(f"No questions found. Cannot generate {filename}.")
        return

    print(f"Generating a larger, more realistic dataset for {filename}...")
    data = []
    column_headers = [f'q_{i+1}' for i in range(len(questions))]

    for _ in range(num_rows):
        answers = [random.choice(['Yes', 'No']) for _ in questions]
        
        # --- MODIFIED LOGIC WITH NOISE ---
        # Base rule
        is_present_base = answers.count('Yes') > len(questions) * 0.4
        
        # Flip the label occasionally to add noise
        if random.random() < 0.15:  # 15% chance to flip the label
            presence = 'Not Present' if is_present_base else 'Present'
        else:
            presence = 'Present' if is_present_base else 'Not Present'
        # --- END OF MODIFIED LOGIC ---

        row = answers + [presence]
        data.append(row)
        
    df = pd.DataFrame(data, columns=column_headers + ['Presence'])
    df.to_csv(filename, index=False)
    print(f"Successfully created {filename} with {num_rows} rows.")

# --- Main script execution ---
disease_questions = get_questions_from_file('data/2.1_Crop_Disease_Characteristics - Sheet1.csv', 4)
insect_questions = get_questions_from_file('data/2.2_Crop_Insect_Characteristics - Sheet1.csv', 3)

generate_training_csv('crop_disease_training_data.csv', disease_questions)
generate_training_csv('crop_insect_training_data.csv', insect_questions)

print("\nProcess complete. You can now retrain your models on the new, improved data.")