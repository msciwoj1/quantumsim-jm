# Gell-Mann matrices
from typing import Literal

import numpy as np

LAMBDA_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
LAMBDA_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
LAMBDA_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
LAMBDA_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
LAMBDA_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
LAMBDA_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
LAMBDA_7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
LAMBDA_8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)

GELL_MANN = (
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
    if index == 0:
        return np.eye(3) * np.sqrt(2 / 3)
    return GELL_MANN[index - 1]
