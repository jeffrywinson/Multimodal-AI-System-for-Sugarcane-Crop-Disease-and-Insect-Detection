# optimize_tabnet_models.py
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna

# --- This objective function is what Optuna will optimize ---
def objective(trial, X_train, y_train, X_val, y_val):
    """
    Defines a single training run for Optuna to evaluate.
    """
    # --- THIS IS THE ONLY NEW LINE ---
    print(f"\n---> Starting Trial #{trial.number}...")
    # --------------------------------
    # 1. Define the Search Space for TabNet Hyperparameters
    # These are the most impactful parameters for TabNet
    mask_type = trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])
    n_da = trial.suggest_int("n_da", 8, 64, step=4) # Width of the decision prediction layer
    n_steps = trial.suggest_int("n_steps", 3, 10)     # Number of decision steps
    gamma = trial.suggest_float("gamma", 1.0, 2.0)    # Relaxation parameter
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)

    # 2. Create and Train the TabNet Model
    clf = TabNetClassifier(
        n_d=n_da, # n_d and n_a are often set to be the same
        n_a=n_da,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        mask_type=mask_type,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr, weight_decay=1e-5),
        seed=42,
        verbose=0 # Set to 0 to keep the logs clean
    )

    # Use a moderate number of epochs for fast trials
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['validation'],
        patience=15, # Stop early if it's not a good trial
        max_epochs=100 
    )

    # 3. Return the score for Optuna to maximize
    score = clf.best_cost
    return score

def find_best_params(csv_path, model_name):
    """
    Loads data and runs the Optuna optimization study.
    """
    print(f"\n--- Starting Optuna Optimization for {model_name} ---")
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    target = 'Presence'
    categorical_features = [col for col in df.columns if col != target]
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])
    X = df.drop(columns=[target]).values
    y = df[target].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=15) # Run 15 experiments

    print(f"--- Optimization Finished for {model_name} ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Validation AUC): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("--- Automated Hyperparameter Optimization for TabNet Models ---")
    
    # Find best params for the insect model
    best_insect_params = find_best_params(
        csv_path='sugarcane_insect_data.csv',
        model_name='tabnet_insect'
    )

    # Find best params for the disease model
    best_disease_params = find_best_params(
        csv_path='sugarcane_deadheart_data.csv',
        model_name='tabnet_disease'
    )
    
    print("\n\n--- OPTIMIZATION COMPLETE ---")
    print("\nBest parameters for tabnet_insect:")
    print(best_insect_params)
    
    print("\nBest parameters for tabnet_disease:")
    print(best_disease_params)
    
    print("\nNext step: Update 'prepare_tabnet_models.py' with these parameters and retrain your final models.")