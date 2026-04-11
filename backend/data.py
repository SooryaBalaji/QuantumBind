import numpy as np
import torch
from chembl_webresource_client.new_client import new_client

def get_chembl_data():
    activity = new_client.activity
    results = activity.filter(
    target_chembl_id='CHEMBL4822',
    standard_type='IC50',
    relation='=',
).only(['molecule_chembl_id', 'standard_value', 'standard_units'])[:200] # filters results to only 200 outputs max instead of all the outputs

    binders = []
    non_binders = []

    for r in results:
        try:
            ic50 = float(r['standard_value'])
            if ic50 < 1000:   # IC50 < 1000nM = binder
                binders.append(ic50)
            else:              # IC50 >= 1000nM = non-binder
                non_binders.append(ic50)
        except:
            continue

    return binders, non_binders

def generate_training_data(n_samples=100):
    print("Fetching real BACE1 binding data from ChEMBL...")
    binders, non_binders = get_chembl_data()
    print(f"Found {len(binders)} binders and {len(non_binders)} non-binders")

    X_quantum = []
    X_protein = []
    y = []

    # Use real IC50 to scale quantum features
    for ic50 in binders[:n_samples//2]:
        affinity = 1 / (1 + ic50/100)  # normalize IC50 to 0-1
        quantum = np.array([
            np.random.normal(-0.89 - affinity*0.1, 0.05),
            np.random.normal(0.54 - affinity*0.05, 0.05),
            np.random.normal(0.04, 0.01),
            np.random.normal(0.0, 0.1),
        ], dtype=np.float32)
        protein = np.random.normal(
            [3924, -8.87, -1.82, -4.65, 21.7, 15.9, 13.4, 119.4, 86.2, 60.9],
            0.1, size=10).astype(np.float32)
        X_quantum.append(quantum)
        X_protein.append(protein)
        y.append(1)

    for ic50 in non_binders[:n_samples//2]:
        affinity = 1 / (1 + ic50/100)
        quantum = np.array([
            np.random.normal(-0.5 + affinity*0.1, 0.1),
            np.random.normal(0.8 + affinity*0.05, 0.1),
            np.random.normal(0.01, 0.005),
            np.random.normal(0.5, 0.2),
        ], dtype=np.float32)
        protein = np.random.normal(
            [3924, -8.87, -1.82, -4.65, 21.7, 15.9, 13.4, 119.4, 86.2, 60.9],
            5.0, size=10).astype(np.float32)
        X_quantum.append(quantum)
        X_protein.append(protein)
        y.append(0)

    return (torch.tensor(np.array(X_quantum)),
            torch.tensor(np.array(X_protein)),
            torch.tensor(y, dtype=torch.float32))

if __name__ == "__main__":
    generate_training_data()