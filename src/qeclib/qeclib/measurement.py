from typing import Literal
from pydantic import Field
from pydantic.dataclasses import dataclass
import uuid


@dataclass()
class Measurement:
    index: int
    number_of_mmts: int
    label: str
    log_qb_id: str | None
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["stabilizer", "log_op", "shrink", "split", "other"] | None = (
        "other"
    )
    related_obj: str | None = None  # Can be used to reference a stabilizer
