from enum import Enum


class HydraEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        return cls[value]
