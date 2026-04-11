import numpy as np
import torch

def generate_training_data(n_samples=100):
    X_quantum = []
    X_protein = []
    y = []

    for _ in range(n_samples // 2):
        # Quantum features close to known BACE1 inhibitor ranges
        quantum = np.array([
            np.random.normal(-0.89, 0.05),  # ground state energy (we got this from vqe.py)
            np.random.normal(0.54, 0.05),  # HOMO-LUMO gap which is typical for the BACE1 inhibitor ranges
            np.random.normal(0.04, 0.01),  # electron correlation (difference btwn HF energy and actual energy)
            np.random.normal(0.0, 0.1),  # dipole moment (H2 is symmetric so dipole is 0)
        ], dtype=np.float32)

        # Protein features that use real BACE1 features with small noise
        protein = np.random.normal([3924, -8.87, -1.82, -4.65, 21.7, 15.9, 13.4, 119.4, 86.2, 60.9],
                                   0.1, size=10).astype(np.float32)
        X_quantum.append(quantum)
        X_protein.append(protein)
        y.append(1)

    for _ in range(n_samples // 2):
        #Different energy profile
        quantum = np.array([
            np.random.normal(-0.5, 0.1),  # higher energy = weaker binder
            np.random.normal(0.8, 0.1),  # larger HOMO-LUMO gap
            np.random.normal(0.01, 0.005),  # less correlation
            np.random.normal(0.5, 0.2),  # different dipole
        ], dtype=np.float32)

        protein = np.random.normal([3924, -8.87, -1.82, -4.65, 21.7, 15.9, 13.4, 119.4, 86.2, 60.9],
                                   5.0, size=10).astype(np.float32)
        X_quantum.append(quantum)
        X_protein.append(protein)
        y.append(0)

    return (torch.tensor(np.array(X_quantum)),
            torch.tensor(np.array(X_protein)),
            torch.tensor(y, dtype=torch.float32))

generate_training_data()