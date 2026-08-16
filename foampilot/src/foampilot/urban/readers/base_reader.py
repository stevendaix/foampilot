from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from foampilot.urban.model.urban_model import UrbanModel


class BaseReader(ABC):
    @abstractmethod
    def read(self, source: Path) -> UrbanModel:
        pass
