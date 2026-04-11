import os
import json
from dotenv import load_dotenv
load_dotenv()

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator

def run_hardware():
    token = os.getenv("IBM_TOKEN")

    # Authenticate
    service = QiskitRuntimeService(
        token=token,
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/0c346df0b2d94325b954da72dba537bc:722e6753-ab55-4df1-b626-2514a97d1dac::"
    )

    backend = service.backend("ibm_kingston")
    print(f"Connected to {backend.name}")

    # H2 Hamiltonian in Pauli form
    hamiltonian = SparsePauliOp.from_list([
        ("IIII", -0.8105),
        ("ZZII", 0.1722),
        ("IIZZ", 0.1722),
        ("ZIZI", 0.1209),
        ("IZIZ", 0.1209),
        ("XXYY", -0.0453),
        ("YYXX", -0.0453),
    ])

    # Simple 4 qubit ansatz
    qc = QuantumCircuit(4)
    qc.x([0, 1])
    qc.ry(0.123826, 2)
    qc.cx(2, 3)
    qc.cx(0, 2)

    # Transpile for hardware
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)

    # Submit job
    estimator = Estimator(backend)
    job = estimator.run([(isa_circuit, isa_hamiltonian)])

    print(f"Job ID: {job.job_id()}")
    print("Waiting for results...")

    result = job.result()
    energy = float(result[0].data.evs)

    print(f"Hardware Energy: {energy:.6f} Hartree")
    print(f"VQE Simulator:   -0.896408 Hartree")
    print(f"ZNE Corrected:   -0.346459 Hartree")

    with open("hardware_results.json", "w") as f:
        json.dump({
            "job_id": job.job_id(),
            "hardware_energy": energy,
            "vqe_energy": -0.896408,
            "zne_energy": -0.346459,
            "backend": "ibm_kingston"
        }, f, indent=2)

    print("Saved to hardware_results.json")
    return energy

if __name__ == "__main__":
    run_hardware()