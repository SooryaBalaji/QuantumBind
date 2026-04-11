import pennylane as qml
from molecule import build_molecule
from pennylane import numpy as np


def run_vqe():
   H, n_qubits = build_molecule()
   n_electrons = 2
   dev = qml.device("default.qubit", wires=n_qubits)


   hf_state = qml.qchem.hf_state(n_electrons, n_qubits)


   singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
   s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)


   params = np.zeros(len(singles) + len(doubles), requires_grad=True)
   optimizer = qml.AdamOptimizer(stepsize=0.1)


   @qml.qnode(dev)
   def circuit(params):
       qml.UCCSD(params, wires=range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
       return qml.expval(H)


   for step in range(10):
       params, energy = optimizer.step_and_cost(circuit, params)


   vqe_energy = energy
   return vqe_energy, params, n_qubits, H, hf_state, s_wires, d_wires



