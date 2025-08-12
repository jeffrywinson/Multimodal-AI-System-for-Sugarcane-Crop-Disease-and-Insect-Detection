import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import argparse

def predict(answers_list):
    encoded_answers = [1 if ans.strip().lower() == 'yes' else 0 for ans in answers_list]
    
    clf = TabNetClassifier()
    clf.load_model('./tabnet/tabnet_insect_model.zip')
    
    # The output from predict() is now an integer representing the class
    prediction_index = clf.predict(np.array([encoded_answers]))[0]
    
    # We need to map the index back to the class name
    # This order comes from how LabelEncoder sorts the strings:
    # 'Early Shoot Borer' (0), 'Internode Borer' (1), 'No Insect' (2)
    class_mapping = {0: "Early Shoot Borer", 1: "Internode Borer", 2: "No Insect"}
    
    predicted_class_name = class_mapping.get(prediction_index, "Unknown")
    
    # We will print the class name. The fusion model will need to handle this.
    print(predicted_class_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict insect presence from answers.")
    parser.add_argument("--answers", type=str, required=True, help="Comma-separated list of Yes/No answers.")
    args = parser.parse_args()
    answer_list = args.answers.split(',')
    predict(answer_list)