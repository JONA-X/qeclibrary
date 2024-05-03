from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from .pauli_op import PauliOp


@dataclass()
class Stabilizer:
    pauli_op: PauliOp
