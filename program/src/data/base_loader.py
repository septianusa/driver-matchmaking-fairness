from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class LoadedData:
    orders: pd.DataFrame
    driver_locations: pd.DataFrame
    driver_scores: pd.DataFrame
    grid_values: pd.DataFrame
    metadata: dict


class BaseDataLoader(ABC):
    """Abstract data-loading boundary used by file and future warehouse adapters."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def load_all(self) -> LoadedData:
        """Load and normalize all datasets required by a simulation run."""

