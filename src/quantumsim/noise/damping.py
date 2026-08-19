# amplitude and phase damping for qubits and qutrits
import numpy as np


def amp_damping_qubit(gamma: float) -> np.ndarray:
    """Return the amplitude damping channel for a qubit."""

    a_0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    a_1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return np.stack([a_0, a_1])


def phase_damping_qubit(gamma: float) -> np.ndarray:
    """Return the phase damping channel for a qubit."""

    a_0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    a_1 = np.array([[0, 0], [0, np.sqrt(gamma)]])
    return np.stack([a_0, a_1])
