from vqe import run_vqe
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer.noise import NoiseModel
import pennylane as qml
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import qibo
from qibo import Circuit
from qibo import gates as qgates
from qibo.noise import NoiseModel as QiboNoiseModel
from qibo.noise import PhaseDampingError

# Use GPU via cupy
qibo.set_backend("qibojit", platform="cupy")


def _pl_to_qibo(n_qubits, params, s_wires, d_wires, hf_state, density_matrix=True):

    with qml.tape.QuantumTape() as tape:
        qml.UCCSD(params, wires=range(n_qubits),
                  s_wires=s_wires, d_wires=d_wires, init_state=hf_state)

    expanded = tape.expand(depth=10, stop_at=lambda op: isinstance(op, (
        qml.RX, qml.RY, qml.RZ,
        qml.PauliX, qml.PauliY, qml.PauliZ,
        qml.Hadamard, qml.CNOT, qml.CZ, qml.SWAP
    )))

    circuit = Circuit(n_qubits, density_matrix=density_matrix)

    for op in expanded.operations:
        name = op.__class__.__name__
        w = op.wires.tolist()
        p = [float(x) for x in op.parameters]

        if   name == 'RX':       circuit.add(qgates.RX(w[0], theta=p[0]))
        elif name == 'RY':       circuit.add(qgates.RY(w[0], theta=p[0]))
        elif name == 'RZ':       circuit.add(qgates.RZ(w[0], theta=p[0]))
        elif name == 'PauliX':   circuit.add(qgates.X(w[0]))
        elif name == 'PauliY':   circuit.add(qgates.Y(w[0]))
        elif name == 'PauliZ':   circuit.add(qgates.Z(w[0]))
        elif name == 'Hadamard': circuit.add(qgates.H(w[0]))
        elif name == 'CNOT':     circuit.add(qgates.CNOT(w[0], w[1]))
        elif name == 'CZ':       circuit.add(qgates.CZ(w[0], w[1]))
        elif name == 'SWAP':     circuit.add(qgates.SWAP(w[0], w[1]))
        else:
            print(f"[warning] Skipping unrecognized gate: {name}")

    return circuit


def _pl_hamiltonian_to_qibo(H, n_qubits):
    from qibo.hamiltonians import SymbolicHamiltonian
    from qibo.symbols import X, Y, Z

    symbol_map = {'PauliX': X, 'PauliY': Y, 'PauliZ': Z}
    terms = []

    for coeff, op in zip(H.coeffs, H.ops):
        name = op.__class__.__name__

        if name in ('Identity', 'I'):
            terms.append(float(coeff))
            continue

        sub_ops = op.operands if hasattr(op, 'operands') else [op]
        term = float(coeff)
        for sub in sub_ops:
            sym_cls = symbol_map.get(sub.__class__.__name__)
            if sym_cls:
                term = term * sym_cls(sub.wires[0])

        terms.append(term)

    return SymbolicHamiltonian(sum(terms), nqubits=n_qubits)


def _run_noisy_circuit(args):
    noise_level, n_qubits, params, s_wires, d_wires, hf_state, H = args

    print(f"[zne] noise={noise_level} starting")

    circuit = _pl_to_qibo(n_qubits, params, s_wires, d_wires, hf_state)

    noise_model = QiboNoiseModel()
    noise_model.add(PhaseDampingError(noise_level))
    noisy_circuit = noise_model.apply(circuit)

    qibo_H = _pl_hamiltonian_to_qibo(H, n_qubits)
    result = noisy_circuit()
    energy = qibo_H.expectation(result.state())

    print(f"[zne] noise={noise_level} done → {float(energy.real):.6f}")
    return float(energy.real)


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

    print("[zne] Running Sherbrooke circuit...")
    sherbrooke_energy = float(sherbrooke_circuit(params))
    print(f"[zne] Sherbrooke done → {sherbrooke_energy:.6f}")

    noise_levels = [0.01, 0.05, 0.1]
    args_list = [
        (noise, n_qubits, params, s_wires, d_wires, hf_state, H)
        for noise in noise_levels
    ]

    print("[zne] Running noise circuits in parallel on GPU...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        energies = list(pool.map(_run_noisy_circuit, args_list))

    coeffs = np.polyfit(noise_levels, energies, 1)
    zne_energy = float(np.polyval(coeffs, 0))

    print(f"[zne] ZNE extrapolated energy → {zne_energy:.6f}")
    return vqe_energy, sherbrooke_energy, zne_energy