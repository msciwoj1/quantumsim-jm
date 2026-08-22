"""Transfer matrix representation of quantum processes."""

from collections.abc import Callable, Sequence

import numpy as np

from ..noise.channel import apply_unitary_channel
from .basis import get_transfer_basis, get_transfer_vector


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


def get_transfer_matrix_from_unitary(
    dims: Sequence[int], unitary: np.ndarray
) -> np.ndarray:
    """Get the transfer matrix corresponding to a unitary.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.
        unitary: Unitary matrix.

    Returns:
        Transfer matrix corresponding to the dimensions.
    """

    def _process(rho: np.ndarray) -> np.ndarray:
        return apply_unitary_channel(rho, unitary)

    return get_transfer_matrix(dims, _process)
