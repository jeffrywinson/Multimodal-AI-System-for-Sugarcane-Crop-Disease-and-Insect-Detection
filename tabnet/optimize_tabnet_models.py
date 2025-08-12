import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Defines a single training run for Optuna to evaluate.
    """
    print(f"\n---> Starting Trial #{trial.number}...")
    
    mask_type = trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])
    n_da = trial.suggest_int("n_da", 24, 64, step=8) # Model size/width
    n_steps = trial.suggest_int("n_steps", 3, 8)     # Number of reasoning steps
    gamma = trial.suggest_float("gamma", 1.2, 2.0)    # A balancing coefficient
    
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    lr = trial.suggest_float("lr", 0.01, 0.03)

    clf = TabNetClassifier(
        n_d=n_da,
        n_a=n_da,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        mask_type=mask_type,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=lr, weight_decay=weight_decay),
        seed=42,
        verbose=0 
    )

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['validation'],
        patience=25, 
        max_epochs=150 
    )

    score = clf.best_cost
    return score

def find_best_params(csv_path, model_name, n_trials=30):
    """
    Loads data and runs the Optuna optimization study.
    """
    print(f"\n--- Starting Optuna Optimization for {model_name} ---")
    
    # Load and prepare data
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

    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    print(f"\n--- Optimization Finished for {model_name} ---")
    trial = study.best_trial
    print(f"  Best Value (Validation AUC): {trial.value}")
    print("  Best Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

if __name__ == "__main__":
    print("--- Automated Hyperparameter Optimization for Final TabNet Models ---")
    best_disease_params = find_best_params(
        csv_path='sugarcane_deadheart_data_r2.csv',
        model_name='tabnet_disease',
        n_trials=30 
    )
    
    print("\n\n--- OPTIMIZATION COMPLETE ---")
    print("\nFinal, best parameters for the disease (Dead Heart) model:")
    print(best_disease_params)
    
    print("\nNext step: Update 'prepare_round2_tabnet.py' with these parameters and run it with --disease_only to train your final model.")