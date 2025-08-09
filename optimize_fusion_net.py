# optimize_fusion_net.py (Corrected for Round 2)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna

# --- Load and prepare the new Round 2 data ---
# This now loads the CSV with 6 input features
df = pd.read_csv('fusion_training_data_r2.csv')
X = df[['yolo_disease', 'yolo_insect', 'tabnet_disease', 'is_esb', 'is_inb', 'is_no_insect']].values
y = df[['is_disease_true', 'is_insect_true']].values

# Split and convert data to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# This "objective" function is what Optuna will try to maximize
# optimize_fusion_net.py (With Tunable Dropout to Prevent Overfitting)

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layers = []
    in_features = 6
    
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 16, 128) # Wider search space
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        
        # --- NEW: Let Optuna choose the best dropout rate ---
        dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
        layers.append(nn.Dropout(p=dropout_rate))
        # --------------------------------------------------
        
        in_features = out_features
    layers.append(nn.Linear(in_features, 2))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop']) # Added AdamW
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    criterion = nn.BCELoss()
    epochs = 200

    for epoch in range(epochs):
        model.train() # Make sure dropout is active
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval() # Disable dropout for evaluation
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()

    return accuracy.item()

# --- Main Optuna Execution ---
if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    
    print("--- Starting Automated Hyperparameter Optimization for Round 2 Fusion Network ---")
    # Let's run 50 trials, which is a good balance for a hackathon
    study.optimize(objective, n_trials=50) 

    print("\n--- Optimization Finished ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value (Accuracy): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))