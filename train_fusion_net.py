# train_fusion_net.py (for Round 2)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

class FusionNetR2(nn.Module):
    def __init__(self):
        super(FusionNetR2, self).__init__()
        # Input layer now takes 6 features
        self.layer_1 = nn.Linear(6, 32)
        self.layer_2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 2) # Still 2 outputs (disease, insect)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# Load data
df = pd.read_csv('fusion_training_data_r2.csv')
X = df[['yolo_disease', 'yolo_insect', 'tabnet_disease', 'is_esb', 'is_inb', 'is_no_insect']].values
y = df[['is_disease_true', 'is_insect_true']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Training Loop
model = FusionNetR2()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 200

print("--- Training Round 2 Fusion Network ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

torch.save(model.state_dict(), 'fusion_model_r2.pth')
print("--- Round 2 fusion model saved as fusion_model_r2.pth ---")