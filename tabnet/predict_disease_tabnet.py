import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import argparse

def predict(answers_list):
    # This encoding must match the training script: 'No' -> 0, 'Yes' -> 1
    encoded_answers = [1 if ans.strip().lower() == 'yes' else 0 for ans in answers_list]

    clf = TabNetClassifier()
    clf.load_model('./tabnet/tabnet_disease_model.zip')

    # Predict probabilities. It returns a value for "Not Present" and "Present". We want the second one.
    prediction_proba = clf.predict_proba(np.array([encoded_answers]))
    probability_present = prediction_proba[0][1]
    print(probability_present)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict disease presence from answers.")
    parser.add_argument("--answers", type=str, required=True, help="Comma-separated list of Yes/No answers.")
    args = parser.parse_args()
    answer_list = args.answers.split(',')
    predict(answer_list)