from pydantic.dataclasses import dataclass

from .pauli_op import PauliOp

CircuitList = list[tuple[str, list[int | tuple[int, int]]]]
Qubit = tuple[int, ...]


@dataclass()
class Stabilizer:
    pauli_op: PauliOp
    anc_qubits: list[
        Qubit
    ]  # List of ancilla qubits that are used for measuring the stabilizer. This might be a single ancilla qubit in the simplest case but can also be a list of ancilla qubits in case more complicated schemes using ancilla cat states or flag qubits is used
    reset: str | None = (
        "reset"  # Whether the ancilla is reset after measurement. Allowed values "reset" or "none". In the future also "conditional" should be supported
    )
