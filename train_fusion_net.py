# train_fusion_net.py (Final Version - Increased Epochs)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define the Fusion Neural Network (same as before)
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.layer_1 = nn.Linear(4, 32)
        self.layer_2 = nn.Linear(32, 16)
        self.layer_3 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# 2. Load Data (same as before)
df = pd.read_csv('fusion_training_data.csv')
X = df[['yolo_disease', 'yolo_insect', 'tabnet_disease', 'tabnet_insect']].values
y = df[['is_disease', 'is_insect']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 3. Training Loop
model = FusionNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

# --- THE ONLY CHANGE IS HERE ---
epochs = 1200 # Was 500

print("--- Final, Deep Training Run for Fusion Network ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluation and Scheduler Step
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    
    scheduler.step(test_loss)

    # Print every 50 epochs to see progress
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

# 4. Save the model
torch.save(model.state_dict(), 'fusion_model.pth')
print("--- Definitive fusion model training complete. Saved as fusion_model.pth ---")