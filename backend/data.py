import numpy as np
import torch
import json
import os

CHEMBL_CACHE = "chembl_cache.json"

FALLBACK_BINDERS = [
    10, 25, 50, 80, 100, 150, 200, 300, 450, 600,
    12, 34, 67, 95, 120, 180, 220, 350, 480, 550,
    15, 40, 75, 90, 130, 170, 250, 320, 410, 580,
    20, 45, 60, 85, 110, 160, 240, 380, 500, 620,
    30, 55, 70, 88, 140, 190, 260, 360, 420, 590,
]
FALLBACK_NON_BINDERS = [
    1200, 1500, 2000, 3000, 5000, 8000, 1100, 1800, 2500, 4000,
    1300, 1600, 2200, 3500, 6000, 9000, 1150, 1900, 2800, 4500,
    1250, 1700, 2100, 3200, 5500, 7000, 1050, 1400, 2300, 3800,
    6500, 8500, 1350, 1550, 2400, 4200, 7500, 1450, 2600, 3900,
    1600, 1800, 2700, 4800, 6800, 9500, 1100, 2900, 3600, 5200,
]

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except Exception as e:
    print(f"[chembl] Import failed ({e}), will use fallback data")
    CHEMBL_AVAILABLE = False


def get_chembl_data():
    if os.path.exists(CHEMBL_CACHE):
        print("[chembl] Loading from cache...")
        with open(CHEMBL_CACHE) as f:
            data = json.load(f)
        return data["binders"], data["non_binders"]

    if CHEMBL_AVAILABLE:
        try:
            print("[chembl] Fetching from ChEMBL API...")
            activity = new_client.activity
            results = activity.filter(
                target_chembl_id='CHEMBL4822',
                standard_type='IC50',
                relation='=',
            ).only(['molecule_chembl_id', 'standard_value', 'standard_units'])[:200]

            binders = []
            non_binders = []
            for r in results:
                try:
                    ic50 = float(r['standard_value'])
                    if ic50 < 1000:
                        binders.append(ic50)
                    else:
                        non_binders.append(ic50)
                except:
                    continue

            with open(CHEMBL_CACHE, "w") as f:
                json.dump({"binders": binders, "non_binders": non_binders}, f)
            print(f"[chembl] Cached {len(binders)} binders, {len(non_binders)} non-binders")
            return binders, non_binders

        except Exception as e:
            print(f"[chembl] API call failed ({e}), using fallback data...")

    print("[chembl] Using fallback data")
    return FALLBACK_BINDERS, FALLBACK_NON_BINDERS


def generate_training_data(n_samples=100):
    binders, non_binders = get_chembl_data()
    print(f"[data] {len(binders)} binders, {len(non_binders)} non-binders")

    X_quantum = []
    X_protein = []
    y = []

    for ic50 in binders[:n_samples // 2]:
        affinity = 1 / (1 + ic50 / 100)
        quantum = np.array([
            np.random.normal(-0.89 - affinity * 0.1, 0.05),
            np.random.normal(0.54 - affinity * 0.05, 0.05),
            np.random.normal(0.04, 0.01),
            np.random.normal(0.0, 0.1),
        ], dtype=np.float32)
        protein = np.random.normal(
            [3924, -8.87, -1.82, -4.65, 21.7, 15.9, 13.4, 119.4, 86.2, 60.9],
            0.1, size=10).astype(np.float32)
        X_quantum.append(quantum)
        X_protein.append(protein)
        y.append(1)

    for ic50 in non_binders[:n_samples // 2]:
        affinity = 1 / (1 + ic50 / 100)
        quantum = np.array([
            np.random.normal(-0.5 + affinity * 0.1, 0.1),
            np.random.normal(0.8 + affinity * 0.05, 0.1),
            np.random.normal(0.01, 0.005),
            np.random.normal(0.5, 0.2),
        ], dtype=np.float32)
        protein = np.random.normal(
            [3924, -8.87, -1.82, -4.65, 21.7, 15.9, 13.4, 119.4, 86.2, 60.9],
            5.0, size=10).astype(np.float32)
        X_quantum.append(quantum)
        X_protein.append(protein)
        y.append(0)

    return (
        torch.tensor(np.array(X_quantum)),
        torch.tensor(np.array(X_protein)),
        torch.tensor(y, dtype=torch.float32)
    )


if __name__ == "__main__":
    generate_training_data()