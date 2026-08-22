# Pauli matrices
from collections.abc import Iterable
from functools import reduce
from typing import Literal

import numpy as np

SIGMA_I = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = (SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z)


def get_pauli(index: Literal[0, 1, 2, 3]) -> np.ndarray:
    """Generate a single qubit Pauli operator, index 0 giving the identity.

    Args:
        index: Index into ``PAULI``, in order "I", "X", "Y", "Z".
    """

    return PAULI[index]


def get_pauli_product(paulistr: Iterable[Literal[0, 1, 2, 3]]) -> np.ndarray:
    """Generate a multi-qubit Pauli operator."""
    return reduce(np.kron, (get_pauli(p) for p in paulistr), np.eye(1, dtype=complex))
