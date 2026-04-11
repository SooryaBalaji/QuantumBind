from vqe import run_vqe
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer.noise import NoiseModel
import pennylane as qml
import numpy as np


def run_vqe():
   backend = FakeSherbrooke()
   noise_model = NoiseModel.from_backend(backend)


   vqe_energy, params, n_qubits, H, hf_state, s_wires, d_wires = run_vqe()
   dev_noisy = qml.device(
       "qiskit.aer",
       wires=n_qubits,
       noise_model=noise_model,
       optimization_level=0,
       # optimization_level=0 prevents qiskit.aer transpiler from performing a pre-execution circuit optimization
   )


   @qml.qnode(dev_noisy)
   def sherbrooke_circuit(params):
       qml.UCCSD(params, wires=range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
       return qml.expval(H)


   sherbrooke_energy = float(sherbrooke_circuit(params))

   energies = []

   noise_levels = [0.01, 0.05, 0.1]

   for noise in noise_levels:
       fcond = qml.noise.wires_in(range(n_qubits))
       noise_op = qml.noise.partial_wires(qml.PhaseDamping, noise)
       pd_noise_model = qml.NoiseModel({fcond: noise_op})


       dev_mixed = qml.device("default.mixed", wires=n_qubits)
       dev_noisy = qml.add_noise(dev_mixed, noise_model=pd_noise_model)


       @qml.qnode(dev_noisy)
       def noisy_circuit(params):
           qml.UCCSD(params, wires=range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
           return qml.expval(H)


       energies.append(float(noisy_circuit(params)))


   coeffs = np.polyfit(noise_levels, energies, 1)
   zne_energy = np.polyval(coeffs, 0)


   return vqe_energy, sherbrooke_energy, zne_energy


print(run_vqe())

