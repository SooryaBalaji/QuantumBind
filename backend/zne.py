from vqe import run_vqe
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer.noise import NoiseModel
import pennylane as qml
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _run_noisy_circuit(args):
    noise, n_qubits, params, s_wires, d_wires, hf_state, H = args

    fcond = qml.noise.wires_in(range(n_qubits))
    noise_op = qml.noise.partial_wires(qml.PhaseDamping, noise)
    pd_noise_model = qml.NoiseModel({fcond: noise_op})

    dev_mixed = qml.device("lightning.mixed", wires=n_qubits)
    dev_noisy = qml.add_noise(dev_mixed, noise_model=pd_noise_model)

    @qml.qnode(dev_noisy)
    def noisy_circuit(params):
        qml.UCCSD(params, wires=range(n_qubits), s_wires=s_wires,
                  d_wires=d_wires, init_state=hf_state)
        return qml.expval(H)

    return float(noisy_circuit(params))


def run_zne():
    vqe_energy, params, n_qubits, H, hf_state, s_wires, d_wires = run_vqe()

    backend = FakeSherbrooke()
    noise_model = NoiseModel.from_backend(backend)
    dev_noisy = qml.device(
        "qiskit.aer",
        wires=n_qubits,
        noise_model=noise_model,
        optimization_level=0,
    )

    @qml.qnode(dev_noisy)
    def sherbrooke_circuit(params):
        qml.UCCSD(params, wires=range(n_qubits), s_wires=s_wires,
                  d_wires=d_wires, init_state=hf_state)
        return qml.expval(H)

    sherbrooke_energy = float(sherbrooke_circuit(params))

    noise_levels = [0.01, 0.05, 0.1]
    args_list = [
        (noise, n_qubits, params, s_wires, d_wires, hf_state, H)
        for noise in noise_levels
    ]

    with ThreadPoolExecutor(max_workers=3) as pool:
        energies = list(pool.map(_run_noisy_circuit, args_list))

    coeffs = np.polyfit(noise_levels, energies, 1)
    zne_energy = np.polyval(coeffs, 0)

    return vqe_energy, sherbrooke_energy, zne_energy


def get_sherbrooke_energy():
    _, sherbrooke_energy, _ = run_zne()
    return sherbrooke_energy