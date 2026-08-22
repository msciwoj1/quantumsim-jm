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


def amp_damping_qutrit_resonator(gamma: float) -> np.ndarray:
    """Return a harmonic resonator-like single parameter amplitude damping channel for a qutrit."""
    a_0 = np.array([[1, 0, 0], [0, np.sqrt(1 - gamma), 0], [0, 0, 1 - gamma]])
    a_1 = np.array(
        [[[0, np.sqrt(gamma), 0], [0, 0, np.sqrt(2 * gamma * (1 - gamma))], [0, 0, 0]]]
    )
    a_2 = np.array([[[0, 0, gamma], [0, 0, 0], [0, 0, 0]]])
    return np.stack([a_0, a_1, a_2])


def relaxation_lindblad_qubit(gamma: float) -> np.ndarray:
    """Return the relaxation Lindblad operator for a qubit."""
    return np.sqrt(gamma) * np.array([[0, 1], [0, 0]])


def dephasing_lindblad_qubit(gamma: float) -> np.ndarray:
    """Return the dephasing Lindblad operator for a qubit."""
    return np.sqrt(2 * gamma) * np.array([[0, 0], [0, 1]])


def relaxation_lindblad_qutrit(gamma: float) -> np.ndarray:
    """Return the relaxation Lindblad operator for a resonator-like qutrit."""
    return np.sqrt(gamma) * np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]])


def dephasing_lindblad_qutrit(gamma: float) -> np.ndarray:
    """Return the dephasing Lindblad operator for a resonator-like qutrit."""
    return np.sqrt(2 * gamma) * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 2]])
