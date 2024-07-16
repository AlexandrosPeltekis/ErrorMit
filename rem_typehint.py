    
import functools
import numpy as np 
import networkx as nx
from qiskit import *
from qiskit.circuit import Gate
from itertools import product
from mitiq import rem, benchmarks, MeasurementResult
from qiskit_aer import AerSimulator
from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString


def execute_rem(circuit, backend=AerSimulator(), shots=100)-> MeasurementResult: 
    result = backend.run(circuit, shots=shots, optimization_level=0, memory=True).result()
    bitstrings = result.get_memory()
    rev_bitstrings = [bitstring[::-1] for bitstring in bitstrings]
    bitstrings_array = np.array([[int(bit) for bit in bitstring] for bitstring in rev_bitstrings])

    return MeasurementResult(bitstrings_array)

def apply_rem(circuit, expected_bs, noise_model=None, shots=10000, p_flip=0.05):

    inverse_confusion_matrix = rem.generate_inverse_confusion_matrix(len(expected_bs), p_flip, p_flip)
    obs = get_projection_operator(expected_bs)


    rem_value = rem.execute_with_rem(
        circuit=circuit, 
        executor=execute_rem, 
        inverse_confusion_matrix=inverse_confusion_matrix, 
        observable=obs
    )

    return rem_value


def get_projection_operator(bitstring):
        n = len(bitstring)
        pauli_terms = []
        
        # Iterate over all possible combinations of 'I' and 'Z' for the length of the bitstring
        for pauli_tuple in product('IZ', repeat=n):
            pauli_string = ''.join(pauli_tuple)
            coeff = 1 / (2 ** n)

            for index, value in enumerate(bitstring):
                if value == 1 and pauli_string[index] == 'Z':
                    coeff *= -1

            pauli_terms.append(PauliString(pauli_string, coeff))
        
        return Observable(*pauli_terms)


def get_circuit_and_bs(layer_depth, n_qubits, seed=None):
    circuit, correct_bitstring = benchmarks.generate_mirror_circuit(
        nlayers=layer_depth,
        two_qubit_gate_prob=1.0,
        connectivity_graph=nx.complete_graph(n_qubits),
        two_qubit_gate_name="CNOT",
        seed=seed,
        return_type="qiskit",
    )        
    return circuit, correct_bitstring



if __name__ == "__main__":
    circ, bs = get_circuit_and_bs(10, 4)
    rem_val = apply_rem(circ, bs)
    print(f"Expectaiton Value After REM: {rem_val}")
