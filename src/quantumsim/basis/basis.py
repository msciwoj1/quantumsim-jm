# general basis utilities
from functools import lru_cache

import numpy as np

from .gell_mann import GELL_MANN
from .pauli import PAULI

SUPPORTED_DIMS: dict[int, tuple[np.ndarray, ...]] = {2: PAULI, 3: GELL_MANN}


@lru_cache
def get_normalized_basis(dim: int) -> np.ndarray:
    """Return the single-qudit operator basis for the given dimension.

    Paulis are used for qubits and Gell-Mann matrices for qutrits, normalized to tr(B_i @ B_i) == 1.

    Args:
        dim: Dimension of the qudit. Can only be 2 or 3.

    Returns:
        Stacked operators of shape (dim**2, dim, dim), normalized so tr(B_i @ B_i) == 1.
    """
    if dim not in SUPPORTED_DIMS:
        raise ValueError(
            f"Unsupported qudit dimension {dim}, expected one of {sorted(SUPPORTED_DIMS)}."
        )
    return np.stack(SUPPORTED_DIMS[dim]) / np.sqrt(2)
