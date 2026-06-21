from __future__ import annotations

from typing import Protocol

import pandas as pd


class SparseCandidateProvider(Protocol):
    sparse_method: str

    def generate_candidates(
        self,
        orders: pd.DataFrame,
        available_drivers: pd.DataFrame,
        config: dict,
    ) -> pd.DataFrame:
        ...

