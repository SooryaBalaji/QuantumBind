import pennylane as qml
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from data import generate_training_data


np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu") #cpu instead of gpu (cuda) since quantum layers only work on cpu

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_layer(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits)}
ql = qml.qnn.TorchLayer(quantum_layer, weight_shapes)

class QuantumBindScreener(nn.Module):
    def __init__(self):
        super(QuantumBindScreener, self).__init__()

        self.quantum_branch = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            ql,  # quantum layer processes the 4 features
            nn.Linear(4, 8),
            nn.Tanh()
        )
        self.protein_branch = nn.Sequential(nn.Linear(10, 8), nn.ReLU()) #10 protein features from protein.py
        self.outputLayer = nn.Linear(16, 1) # 8 output from protein + 8 outputs from quantum and produces one binding result between 0 and 1

    def forward(self, quantum_features, protein_features):
        q = self.quantum_branch(quantum_features) # seperate into quantum features
        p = self.protein_branch(protein_features) # seperate into protein features
        combined = torch.cat([q, p], dim=1) # combines both tensors
        return torch.sigmoid(self.outputLayer(combined)) # once again squashes the value between 0 and 1 for extra measure

model = QuantumBindScreener().to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

X_quantum, X_protein, y = generate_training_data()

# Normalize both inputs
X_quantum = (X_quantum - X_quantum.mean(dim=0)) / (X_quantum.std(dim=0) + 1e-8)
X_protein = (X_protein - X_protein.mean(dim=0)) / (X_protein.std(dim=0) + 1e-8)

X_q_train, X_q_test, X_p_train, X_p_test, y_train, y_test = train_test_split(
    X_quantum.numpy(), X_protein.numpy(), y.numpy(),
    test_size=0.2, random_state=42
)

X_q_train = torch.tensor(X_q_train)
X_q_test = torch.tensor(X_q_test)
X_p_train = torch.tensor(X_p_train)
X_p_test = torch.tensor(X_p_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(X_q_train, X_p_train).squeeze()
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "../screener_model.pth")

model.eval()
with torch.no_grad():
    test_predictions = model(X_q_test, X_p_test).squeeze()
    predicted_labels = (test_predictions > 0.5).float()
    accuracy = (predicted_labels == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
