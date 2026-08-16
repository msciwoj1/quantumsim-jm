# Pauli matrices
from collections.abc import Iterable
from typing import Literal

import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]])
SIGMA_Y = np.array([[0, -1j], [1j, 0]])
SIGMA_Z = np.array([[1, 0], [0, -1]])


def get_pauli(pauli: Literal["X", "Y", "Z", "I"]) -> np.ndarray:
    """Generate a single qubit Pauli operator."""
    if pauli == "X":
        return SIGMA_X
    if pauli == "Y":
        return SIGMA_Y
    if pauli == "Z":
        return SIGMA_Z
    return np.eye(2)


def get_pauli_product(paulistr: Iterable[Literal["X", "Y", "Z", "I"]]) -> np.ndarray:
    """Generate a multi-qubit Pauli operator."""
    pauli = 1
    for p in paulistr:
        pauli = np.kron(pauli, get_pauli(p))
    return pauli.astype(complex)
