from __future__ import annotations
from typing import Optional, Literal
from pydantic import Field
from pydantic.dataclasses import dataclass
import uuid


@dataclass()
class Measurement():
    index: int
    number_of_mmts: int
    label: str
    log_qb_id: str
    uuid: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Optional[Literal["stabilizer", "log_op", "shrink", "split", "other"]] = "other"
    related_obj: Optional[str] = None # Can be used to reference a stabilizer
