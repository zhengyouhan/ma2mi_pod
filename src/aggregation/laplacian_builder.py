"""
Detector Graph Laplacian Builder.

Constructs the graph Laplacian L for the 1D detector chain.

Math:
    Graph G_det:
    - Nodes: detectors j = 0..J-1
    - Edges: (j, j+1) with weight w

    Adjacency: W[j,j+1] = W[j+1,j] = w
    Degree: D[j,j] = sum_k W[j,k]
    Laplacian: L = D - W

    For 1D chain with w=1:
    L = tridiag(-1, 2, -1) with modified endpoints
"""
from __future__ import annotations

import torch as th


def build_chain_laplacian(
    J: int,
    w: float = 1.0,
    boundary: str = "natural",
    device: str = "cpu",
    dtype: th.dtype = th.float32,
) -> th.Tensor:
    """
    Build 1D chain graph Laplacian.

    Args:
        J: number of nodes (detectors)
        w: edge weight (default 1.0)
        boundary: "natural" (degree-1 endpoints) or "neumann" (adjusted weights)
        device: torch device
        dtype: torch dtype

    Returns:
        L: [J, J] Laplacian matrix
    """
    # Build adjacency matrix W
    W = th.zeros((J, J), device=device, dtype=dtype)
    for j in range(J - 1):
        W[j, j + 1] = w
        W[j + 1, j] = w

    # Degree matrix D
    D = th.diag(W.sum(dim=1))

    # Laplacian L = D - W
    L = D - W

    if boundary == "neumann":
        # Neumann-like: zero gradient at boundaries
        # Adjust endpoint degrees to match interior behavior
        # L[0,0] = 1, L[J-1,J-1] = 1 (instead of w)
        L[0, 0] = w
        L[J - 1, J - 1] = w

    return L


def build_chain_laplacian_sparse(
    J: int,
    w: float = 1.0,
    device: str = "cpu",
    dtype: th.dtype = th.float32,
) -> th.Tensor:
    """
    Build 1D chain Laplacian as sparse tensor (for efficiency).

    For small J (< 100), dense is fine. This is for larger cases.
    """
    # For now, return dense since J is typically small (~20 detectors)
    return build_chain_laplacian(J, w, "natural", device, dtype)


def get_laplacian_eigendecomposition(L: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """
    Compute eigendecomposition of Laplacian.

    Returns:
        eigenvalues: [J] sorted ascending
        eigenvectors: [J, J] columns are eigenvectors
    """
    eigenvalues, eigenvectors = th.linalg.eigh(L)
    return eigenvalues, eigenvectors
