# optimize_fusion_net.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna

# Load and prepare data once to be efficient
df = pd.read_csv('fusion_training_data.csv')
X = df[['yolo_disease', 'yolo_insect', 'tabnet_disease', 'tabnet_insect']].values
y = df[['is_disease', 'is_insect']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# This "objective" function is what Optuna will try to maximize
def objective(trial):
    # --- 1. Define the Search Space for Hyperparameters ---
    n_layers = trial.suggest_int('n_layers', 2, 4) # Try between 2 and 4 layers
    layers = []
    in_features = 4
    for i in range(n_layers):
        # Suggest number of neurons for each layer
        out_features = trial.suggest_int(f'n_units_l{i}', 8, 64) 
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 2))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)

    # Suggest an optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    # Suggest a learning rate
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    criterion = nn.BCELoss()
    epochs = 400 # Use a moderate number of epochs for each trial to save time

    # --- 2. Training and Evaluation Loop (for a single trial) ---
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # --- 3. Evaluate and Return the Score ---
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()

    return accuracy.item()


# --- Main Optuna Execution ---
if __name__ == "__main__":
    # Create a "study" to maximize the accuracy
    study = optuna.create_study(direction='maximize')
    
    # Start the optimization. n_trials is how many different combinations to test.
    print("--- Starting Automated Hyperparameter Optimization ---")
    study.optimize(objective, n_trials=100) # Let's run 100 experiments

    print("\n--- Optimization Finished ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value (Accuracy): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))