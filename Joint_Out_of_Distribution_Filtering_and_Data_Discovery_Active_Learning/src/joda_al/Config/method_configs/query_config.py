from dataclasses import dataclass, field
from typing import Dict

from joda_al.Config.config_def import NestedConfig


@dataclass
class QueryMethodConfig(NestedConfig):
    method_type: str = "lloss"
    args: Dict = field(default_factory=lambda: {})
