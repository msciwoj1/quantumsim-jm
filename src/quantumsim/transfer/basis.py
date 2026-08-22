"""Transfer basis construction and conversion between density matrices and vectors."""

from collections.abc import Sequence
from functools import lru_cache, reduce
from itertools import product

import numpy as np

from quantumsim.basis import GELL_MANN, PAULI

SUPPORTED_DIMS: dict[int, tuple[np.ndarray, ...]] = {2: PAULI, 3: GELL_MANN}


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


@lru_cache
def get_transfer_basis(dims: tuple[int, ...]) -> np.ndarray:
    """Return the transfer basis for a given list of dimensions.

    Each qudit contributes its own normalized single-qudit basis (Paulis for qubits,
    Gell-Mann matrices for qutrits).

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.

    Returns:
        Stacked Pauli/Gell-Mann operators in the transfer basis, normalized to tr(B_i @ B_i) == 1.
    """
    per_qudit = [get_normalized_basis(dim) for dim in dims]
    return np.stack(
        [
            reduce(np.kron, operators, np.eye(1, dtype=complex))
            for operators in product(*per_qudit)
        ]
    )


def get_transfer_vector(dims: Sequence[int], rho: np.ndarray) -> np.ndarray:
    """Return a vector in a transfer basis for a given density matrix.

    The transfer basis is normalized so that the trace of the square of each matrix is 1. This allows
    extracting the vector by simple matrix multiplication and extraction of the traces by Einsum, ei. the resulting
    vector is a 1D array of tr(basis[i] @ rho), just done more efficiently.

    If more dims are present in the density matrix, they are transposed to the end. This allows easy creation
    of the transfer matrix.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.
        rho: Density matrix, dimension needs to be a product of all the dims.

    Returns:
        Vector in a transfer basis corresponding to the density matrix.
    """
    basis = get_transfer_basis(tuple(dims))
    return np.einsum("ijk,...kj->i...", basis, rho)


def get_density_matrix(dims: Sequence[int], vector: np.ndarray) -> np.ndarray:
    """Return a density matrix from a vector in a transfer basis.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.
        vector: Vector in a transfer basis.

    Returns:
        Density matrix corresponding to the vector.
    """
    basis = get_transfer_basis(tuple(dims))
    return np.tensordot(vector, basis, axes=1)
