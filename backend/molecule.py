from pennylane import qchem
import numpy as np


def build_molecule():
   symbols =  ["H", "H"]


   coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


   H, n_qubits = qchem.molecular_hamiltonian(symbols, coordinates, basis="sto-3g")


   return H, n_qubits

