from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from .logical_qubit import LogicalQubit


@dataclass()
class Processor:
    name: str
    logical_qubits: List[LogicalQubit]

    def print_logical_qubits(self):
        for qb in self.logical_qubits:
            print(qb.id)

    def is_log_qb_id_contained(self, id: str):
        for qb in self.logical_qubits:
            if id == qb.id:
                return True
        return False

    def add_logical_qubit(self, logical_qubit: LogicalQubit):
        if self.is_log_qb_id_contained(id):
            raise ValueError("Logical qubit id already exists on the processor")

        self.logical_qubits.append(logical_qubit)
