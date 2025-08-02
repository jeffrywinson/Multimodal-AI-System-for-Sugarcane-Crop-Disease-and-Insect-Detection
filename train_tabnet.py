# train_tabnet.py (Updated for Experimentation)
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

def train(csv_path, model_name, epochs, patience, lr): # Added more arguments
    """Trains and saves a TabNet model with more controls."""
    print(f"--- Starting training for {model_name} with epochs={epochs}, patience={patience}, lr={lr} ---")

    df = pd.read_csv(csv_path)
    target = 'Presence'
    
    # Preprocessing (same as before)
    categorical_features = [col for col in df.columns if col != target]
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # We can define the optimizer here to control the learning rate (lr)
    optimizer_params = dict(lr=lr, weight_decay=1e-5)

    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=optimizer_params,
        verbose=1,
        seed=42,
        device_name='cpu' 
    )

    print("Training the model...")
    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=['validation'],
        patience=patience, # Use patience from arguments
        max_epochs=epochs  # Use epochs from arguments
    )

    saved_model_path = clf.save_model(f'./{model_name}_model')
    print(f"--- Training finished. Model saved at {saved_model_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TabNet model.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate.") # Default is 0.02
    
    args = parser.parse_args()
    train(args.csv_path, args.model_name, args.epochs, args.patience, args.lr)