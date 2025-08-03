# predict_disease_tabnet.py
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import argparse

def predict(answers_list):
    # This encoding must match the training script: 'No' -> 0, 'Yes' -> 1
    encoded_answers = [1 if ans.strip().lower() == 'yes' else 0 for ans in answers_list]

    clf = TabNetClassifier()
    clf.load_model('./tabnet_disease_model.zip')

    prediction = clf.predict(np.array([encoded_answers]))

    # This encoding must match the training script: 'Not Present' -> 0, 'Present' -> 1
    result = 'Present' if prediction[0] == 1 else 'Not Present'
    print(result) # Print only the final result for the shell script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict disease presence from answers.")
    parser.add_argument("--answers", type=str, required=True, help="Comma-separated list of Yes/No answers.")
    args = parser.parse_args()
    answer_list = args.answers.split(',')
    predict(answer_list)