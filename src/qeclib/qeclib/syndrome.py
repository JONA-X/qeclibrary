from typing import Literal
from pydantic import Field
from pydantic.dataclasses import dataclass
import uuid

from .measurement import Measurement


@dataclass()
class Syndrome:
    log_qb_id: str
    qec_cycle: int
    stab_mmt_ids: list[str]
    stabilizer: str
