from typing import Literal
from pydantic import Field
from pydantic.dataclasses import dataclass
import uuid


@dataclass()
class StabilizerMeasurement:
    log_qb_id: str
    qec_cycle: int
    mmt_uuids: list[tuple[str, int]]
    stabilizer: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
