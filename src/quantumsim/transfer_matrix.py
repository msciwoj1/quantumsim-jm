# Process transfer matrix utilities
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import product
from typing import Literal

import numpy as np

from .gell_mann import get_gell_mann
from .pauli import get_pauli

INDEX_TO_PAULI: dict[Literal[0, 1, 2, 3], Literal["I", "X", "Y", "Z"]] = {
    0: "I",
    1: "X",
    2: "Y",
    3: "Z",
}


@lru_cache
def get_transfer_basis(dims: tuple[int]) -> np.ndarray:
    """Return the transfer basis for a given list of dimensions.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.

    Returns:
        List of Pauli/Gell-Mann operators in the transfer basis.
    """
    basis = []
    for idx in product(*[range(dim**2) for dim in dims]):
        matrix = 1
        for j, k in enumerate(idx):
            if dims[j] == 2:
                matrix = np.kron(matrix, get_pauli(INDEX_TO_PAULI[k])) / np.sqrt(2)
            else:
                matrix = np.kron(matrix, get_gell_mann(k)) / np.sqrt(2)
        basis.append(matrix)
    return np.stack(basis)


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


def get_transfer_matrix(
    dims: Sequence[int], process: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Return a transfer matrix corresponding to the given quantum process.

    Apply the process to each basis matrix, obtaining a matrix. Convert the resulting matrices to vectors
    in the transfer basis. The vectors are then combined into a transfer matrix.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.
        process: Quantum process to be represented as a transfer matrix, represented as a function that takes
            a density matrix and returns a density matrix. Assume it is vectorized, and that it is linear.

    Returns:
        Transfer matrix corresponding to the dimensions.
    """
    basis = get_transfer_basis(tuple(dims))
    return get_transfer_vector(dims, process(basis))
