from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class ComparisonCase:
    lambda_driver_score: float
    matching_algorithm: str
    sparse_method: str

    @property
    def variant_id(self) -> str:
        lambda_label = str(self.lambda_driver_score).replace(".", "p")
        return (
            f"lambda={lambda_label}__solver={self.matching_algorithm}"
            f"__sparse={self.sparse_method}"
        )


def expand_comparison_cases(config: dict) -> list[ComparisonCase]:
    comparison = config.get("comparison", {})
    lambdas = comparison.get("lambda_driver_score_values", [0.0, 0.1, 0.2, 0.3, 0.4])
    algorithms = comparison.get("matching_algorithms", ["hungarian", "greedy", "auction"])
    sparse_methods = comparison.get("sparse_methods", ["bfs_h3", "a2gat"])
    return [
        ComparisonCase(float(lambda_value), str(algorithm), str(sparse_method))
        for lambda_value, algorithm, sparse_method in product(lambdas, algorithms, sparse_methods)
    ]

