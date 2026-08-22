"""Embedded-qubit (leakage) analysis of transfer matrices.

A qutrit whose lowest two levels are used as a qubit is described in the "embedded qubit"
basis, in which the Gell-Mann operators lambda_0 and lambda_8 are replaced by the embedded
qubit identity diag(1, 1, 0) and its complement 1/sqrt(2) diag(0, 0, 1). Rotating a transfer
matrix into that basis separates the qubit subspace from the leakage level.
"""

from collections.abc import Sequence

import numpy as np

from .process import get_transfer_matrix_from_unitary


def get_gell_mann_to_embedded_qubit_basis() -> np.ndarray:
    """Generate a basis change matrix from Gell-Mann to the embedded qubit representation of a qutrit.

    Only the lambda_0/lambda_8 block is mixed.
    """
    basis_change = np.eye(9)
    basis_change[0, 0] = np.sqrt(2 / 3)
    basis_change[0, 8] = np.sqrt(1 / 3)
    basis_change[8, 0] = np.sqrt(1 / 3)
    basis_change[8, 8] = -np.sqrt(2 / 3)
    return basis_change


def get_embedded_qubit_basis_mask() -> np.ndarray:
    """Generate a mask for the embedded qubit representation of a qutrit.

    Keeps the embedded qubit block and drops lambda_4-7 along with the leakage
    complement that replaces lambda_8.
    """
    mask = np.diag([1, 1, 1, 1, 0, 0, 0, 0, 0])
    return mask


def _kron_per_dim(
    dims: Sequence[int],
    qubit_block: np.ndarray | None = None,
    qutrit_block: np.ndarray | None = None,
) -> np.ndarray:
    """Tensor a per-qudit block over dims, contributing the identity for every qubit.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.
        qutrit_block: 9x9 block contributed by each qutrit.

    Returns:
        Register-wide operator acting on transfer vectors.
    """
    qubit_block = np.eye(4) if qubit_block is None else qubit_block
    qutrit_block = np.eye(9) if qutrit_block is None else qutrit_block
    out = np.atleast_2d(1)
    for dim in dims:
        out = np.kron(out, qubit_block if dim == 2 else qutrit_block)
    return out


def get_rotated_qubit_transfer_matrix(
    dims: Sequence[int], transfer_matrix: np.ndarray
) -> np.ndarray:
    """Rotate the transfer matrix to match the embedded qubit basis.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.
        transfer_matrix: Transfer matrix in Pauli/Gell-Mann basis.

    Returns:
        Transfer matrix in Pauli/leakage Pauli basis.
    """
    full_basis_change = _kron_per_dim(
        dims, qutrit_block=get_gell_mann_to_embedded_qubit_basis()
    )
    trans_basis_change = np.transpose(full_basis_change)

    return np.einsum(
        "ij,...jk, km->...im", full_basis_change, transfer_matrix, trans_basis_change
    )


def get_full_qubit_trace_mask(dims: Sequence[int]) -> np.ndarray:
    """Return a mask that can be used to calculate the entanglement fidelity for only qubit levels."""
    return _kron_per_dim(dims, qutrit_block=get_embedded_qubit_basis_mask())


def get_qubit_ent_fidelity_leakage(
    dims: Sequence[int],
    transfer_matrix: np.ndarray,
    ideal_unitary: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Calculate the entanglement fidelity, leakage rate and average gate fidelity of the process assuming it applies to qubits.

    First rotate the transfer matrix to the Pauli/leakage Pauli basis, in which Gell-Mann operators lambda_0
    and lambda_8 are replaced by the embedded qubit identity diag(1, 1, 0) and its complement 1/sqrt(2) diag(0, 0, 1).
    Then, the entanglement fidelity is simply the trace of the rotated transfer matrix over the embedded qubit part of
    the basis, ie. ignoring lambda_4-7 and the aforementioned complement, divided by the dimension of the
    qubit basis, ie. 2**len(dims).

    The leakage rate is 1 - the qubit identity component, and the average gate fidelity is taken from
    Wood and Gambetta (arxiv.org/abs/1704.03081):

    .. math:

        F = \frac{d_1F_e + 1 - L}{d_1 + 1}

    Where d_1 is the dimension of the qubit basis, F_e is the entanglement fidelity and L is the leakage rate.

    Args:
        dims: Tuple of dimensions for each qudit. Can only contain 2 or 3.
        transfer_matrix: Transfer matrix in Pauli/Gell-Mann basis.
        ideal_unitary: Ideal unitary for the gate.

    Returns:
        Entanglement fidelity, leakage and average gate fidelity.
    """
    if ideal_unitary is not None:
        inverse_ideal_tm = get_transfer_matrix_from_unitary(
            dims, ideal_unitary.conj().T
        )
        tm = np.einsum("...ij, jk", transfer_matrix, inverse_ideal_tm)
    else:
        tm = transfer_matrix
    qubit_dimension = 2 ** len(dims)
    rotated = get_rotated_qubit_transfer_matrix(dims, tm)
    leakage = 1 - rotated[..., 0, 0]
    mask = get_full_qubit_trace_mask(dims)
    ent_fidelity = 1 / qubit_dimension**2 * np.einsum("ij,...ji->...", mask, rotated)
    av_gate_fidelity = (qubit_dimension * ent_fidelity + 1 - leakage) / (
        qubit_dimension + 1
    )
    return ent_fidelity, leakage, av_gate_fidelity
