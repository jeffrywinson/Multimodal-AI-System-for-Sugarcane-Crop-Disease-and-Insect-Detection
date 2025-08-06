# train_fusion_net.py (Definitive Version - Using Best Parameters from Trial 22)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split

# --- STEP 1: DEFINE THE WINNING ARCHITECTURE FOUND BY OPTUNA ---
# These are the exact, best parameters from your Trial 22 log.
BEST_PARAMS = {
    'n_layers': 3,
    'n_units_l0': 45,
    'n_units_l1': 60,
    'n_units_l2': 55,
    'optimizer': 'Adam',
    'lr': 0.00626330958689155
}
# ---------------------------------------------------------------

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        layers = []
        in_features = 4
        # Dynamically build the network based on the best params
        for i in range(BEST_PARAMS['n_layers']):
            out_features = BEST_PARAMS[f'n_units_l{i}']
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, 2))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 2. Load Data (same as before)
df = pd.read_csv('fusion_training_data.csv')
X = df[['yolo_disease', 'yolo_insect', 'tabnet_disease', 'tabnet_insect']].values
y = df[['is_disease', 'is_insect']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 3. Training Loop with the Best Parameters
model = FusionNet()
criterion = nn.BCELoss()
optimizer_class = getattr(optim, BEST_PARAMS['optimizer'])
optimizer = optimizer_class(model.parameters(), lr=BEST_PARAMS['lr'])
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=15)

# Give it plenty of time to train to its peak
epochs = 1500 

print(f"--- Training Final Model with Best Parameters from Optuna ---")
print(f"Parameters: {BEST_PARAMS}")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    
    scheduler.step(test_loss)

    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

# 4. Save the final, best model
torch.save(model.state_dict(), 'fusion_model.pth')
print("--- Definitive fusion model training complete. Saved as fusion_model.pth ---")