# Gell-Mann matrices
from typing import Literal

import numpy as np


# The identity element is scaled so that tr(LAMBDA_0 @ LAMBDA_0) == 2.
LAMBDA_0 = np.eye(3, dtype=complex) * np.sqrt(2 / 3)
LAMBDA_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
LAMBDA_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
LAMBDA_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
LAMBDA_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
LAMBDA_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
LAMBDA_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
LAMBDA_7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
LAMBDA_8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)

GELL_MANN = (
    LAMBDA_0,
    LAMBDA_1,
    LAMBDA_2,
    LAMBDA_3,
    LAMBDA_4,
    LAMBDA_5,
    LAMBDA_6,
    LAMBDA_7,
    LAMBDA_8,
)


def get_gell_mann(index: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]) -> np.ndarray:
    """Generate a single qutrit Gell-Mann operator, index 0 giving the identity."""
    return GELL_MANN[index]
