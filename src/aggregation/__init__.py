"""
Aggregation module: Micro → Macro via Projection + Graph Filter.
"""
from src.aggregation.projection import (
    hard_binning_projection,
    soft_assignment_projection,
    fill_empty_cells,
)
from src.aggregation.laplacian_builder import (
    build_chain_laplacian,
    get_laplacian_eigendecomposition,
)
from src.aggregation.graph_filter import (
    tikhonov_filter,
    tikhonov_filter_2d,
    GraphAggregator,
)

__all__ = [
    "hard_binning_projection",
    "soft_assignment_projection",
    "fill_empty_cells",
    "build_chain_laplacian",
    "get_laplacian_eigendecomposition",
    "tikhonov_filter",
    "tikhonov_filter_2d",
    "GraphAggregator",
]
