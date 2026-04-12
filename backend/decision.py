from data import generate_training_data
from backend.screener import model, device
import torch

_X_q, _X_p, _ = generate_training_data()
Q_MEAN, Q_STD = _X_q.mean(dim=0).to(device), (_X_q.std(dim=0) + 1e-8).to(device)
P_MEAN, P_STD = _X_p.mean(dim=0).to(device), (_X_p.std(dim=0) + 1e-8).to(device)

def predict_binding(vqe_energy, zne_energy):
    v1 = float(vqe_energy)
    v2 = float(zne_energy)
    quantum_features = torch.tensor([[v1, v2, 0.0, 0.0]], dtype=torch.float32).to(device)

    from backend.protein import get_protein_features
    protein_raw = torch.tensor(
        get_protein_features(), dtype=torch.float32
    ).unsqueeze(0).to(device)

    X_quantum_train, X_protein_train, _ = generate_training_data()
    X_quantum_train = X_quantum_train.to(device)
    X_protein_train = X_protein_train.to(device)

    # Normalize using training data stats
    #Normalization formula:
    #normalized = (value - mean) / std

    quantum_features = (quantum_features - X_quantum_train.mean(dim=0)) / (X_quantum_train.std(dim=0) + 1e-8)
    protein_features = (protein_raw - X_protein_train.mean(dim=0)) / (X_protein_train.std(dim=0) + 1e-8)

    model.eval()
    with torch.no_grad():
        score = model(quantum_features, protein_features).item()

    print(f"Binding Score: {score:.4f}")
    if score > 0.5:
        print("Decision: Worth pursuing ")
    else:
        print("Decision: Reject ")
    return score

def the_decision(vqe_energy, zne_energy):
    return predict_binding(vqe_energy, zne_energy)
