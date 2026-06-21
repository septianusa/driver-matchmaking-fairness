from __future__ import annotations

import os
from pathlib import Path

from src.data.base_loader import BaseDataLoader, LoadedData


class MaxComputeDataLoader(BaseDataLoader):
    """Future adapter boundary for Alibaba Cloud MaxCompute.

    Expected implementation steps:
    1. Read credentials from environment variables in `.env.example`.
    2. Read SQL templates from the `sql/` directory.
    3. Bind `simulation_date` safely through the MaxCompute client.
    4. Return pandas DataFrames with the raw schemas expected by `FileDataLoader`'s
       normalization layer, or factor that normalization into a shared adapter helper.

    Secrets must never be hard-coded in the repository.
    """

    SQL_FILES = {
        "orders": "orders.sql",
        "driver_locations": "driver_locations.sql",
        "driver_scores": "driver_scores.sql",
        "grid_values": "grid_values.sql",
    }

    def _read_sql(self, name: str) -> str:
        root = Path(self.config.get("_project_root", "."))
        return (root / "sql" / self.SQL_FILES[name]).read_text(encoding="utf-8")

    def load_all(self) -> LoadedData:
        required_env = [
            "MAXCOMPUTE_ACCESS_ID",
            "MAXCOMPUTE_ACCESS_KEY",
            "MAXCOMPUTE_PROJECT",
            "MAXCOMPUTE_ENDPOINT",
        ]
        missing = [key for key in required_env if not os.getenv(key)]
        if missing:
            raise NotImplementedError(
                "MaxCompute loader is not implemented yet and is missing environment variables: "
                + ", ".join(missing)
            )
        raise NotImplementedError(
            "MaxCompute integration is intentionally separated from simulation logic. "
            "Implement client authentication, SQL binding, execution, and DataFrame conversion here."
        )

