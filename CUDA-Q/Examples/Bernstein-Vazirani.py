# The Bernstein-Vazirani algorithm aims to identify the bitstring encoded in a given function.

# here, we generate a random bitstring and encode it into a inner-product oracle, and define a kernel for the Bernstein-Vazirani algorithm.
# Then, we simulate the kernel and return the most probable bitstring from its execution.

# If all goes well, the state measured with the highest probability should be our hidden bitstring!

import cudaq
import random

from typing import List

def random_bits(length: int) -> List[int]:
    """Generate a random bitstring of given length."""
    bitset = []
    for _ in range(length):
        bitset.append(random.randint(0, 1))
    return bitset

qubit_counts = 5
cudaq.set_target('nvidia')

hidden_bits = random_bits(qubit_counts)
print(f'Hidden bitstring: {hidden_bits}')

@cudaq.kernel
def oracle(register: cudaq.qview, auxiliary_qubit: cudaq.qubit, hidden_bits: List[int]):
    for index, bit in enumerate(hidden_bits):
        if bit == 1:
            # apply CX, current qubit as control, auxiliary as target
            x.ctrl(register[index], auxiliary_qubit)

@cudaq.kernel
def bernstein_vazirani(hidden_bits: List[int]):
    # allocate the specified number of qubits that corresponds to the length of the hidden bitstring
    qubits = cudaq.qvector(len(hidden_bits))
    auxiliary = cudaq.qubit()

    # prepare the auxiliary qubit in |-> state
    h(auxiliary)
    z(auxiliary)

    # place the rest of register in a superposition
    h(qubits)

    # Query the oracle
    oracle(qubits, auxiliary, hidden_bits)

    # Apply inverse Hadamard to the qubits
    h(qubits)

    mz(qubits)

print(cudaq.draw(bernstein_vazirani, hidden_bits))
result = cudaq.sample(bernstein_vazirani, hidden_bits)

print(f"encoded bitstring = {hidden_bits}")
print(f"measured state = {result.most_probable()}")
print(
    f"Were we successful? {''.join([str(i) for i in hidden_bits]) == result.most_probable()}"
    )
# it is application of phase kickback