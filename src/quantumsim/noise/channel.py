# helpers for noise channels
import numpy as np


def multiply_kraus_channels(kraus1: np.ndarray, kraus2: np.ndarray) -> np.ndarray:
    """Multiply Kraus channels on the same qudit."""
    prod = np.einsum("ijk,mkl->imjl", kraus1, kraus2)
    shape1 = kraus1.shape
    shape2 = kraus2.shape
    return prod.reshape(shape1[0] * shape2[0], shape1[1], shape1[2])


def kron_kraus_channels(kraus1: np.ndarray, kraus2: np.ndarray) -> np.ndarray:
    """Kronecker product of Kraus channels."""
    k, n, _ = kraus1.shape
    l, m, _ = kraus2.shape
    out = np.einsum("kij,lpq->klipjq", kraus1, kraus2)
    return out.reshape(k * l, n * m, n * m)


def extend_kraus_channel(
    kraus: np.ndarray, idx: int, dims: tuple[int, ...]
) -> np.ndarray:
    """Extend a single Kraus channel to act on multiple qudits."""
    dims_before = dims[:idx]
    dims_after = dims[idx + 1 :]
    first_half = kron_kraus_channels(
        np.stack([np.eye(int(np.prod(dims_before)))]), kraus
    )
    full = kron_kraus_channels(first_half, np.stack([np.eye(int(np.prod(dims_after)))]))
    return full


def apply_kraus_channel(rho: np.ndarray, kraus: np.ndarray) -> np.ndarray:
    return np.einsum("ijk,...kl,iml->...jm", kraus, rho, kraus.conj())


def apply_unitary_channel(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...jk,lk->...il", unitary, rho, unitary.conj())


def eval_lindbladian(
    rho: np.ndarray,
    hamiltonian: np.ndarray | None = None,
    lindbladian: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the Lindbladian superoperator for the given density matrix.

    Args:
        rho: The density matrix, possibly stacked on the first dimension.
        hamiltonian: The Hamiltonian in matrix form.
        lindbladian: The Lindbladian jump operators in matrix form, stacked on the first dimension.
    """
    res = np.zeros_like(rho)
    if hamiltonian is not None:
        ham_part = 1j * np.einsum("...ij,jk->...ik", rho, hamiltonian) - 1j * np.einsum(
            "...ij,ki->...kj", rho, hamiltonian
        )
        res += ham_part

    conj_lindbladian = lindbladian.conj()
    if lindbladian is not None:
        lind_one = np.einsum(
            "mki, ...ij, mlj->...kl", lindbladian, rho, conj_lindbladian
        )
        lind_two = np.einsum(
            "mik, mij, ...jl->...kl", conj_lindbladian, lindbladian, rho
        )
        lind_three = np.einsum(
            "...ki, mji, mjl->...kl", rho, conj_lindbladian, lindbladian
        )
        res += lind_one - 1 / 2 * lind_two - 1 / 2 * lind_three
    return res


def apply_lindbladian_rk4(
    rho: np.ndarray,
    timestep: float,
    hamiltonian: np.ndarray | None = None,
    lindbladian: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the Lindbladian superoperator to the given density matrix using the fourth-order Runge-Kutta method.

    Args:
        rho: The density matrix, possibly stacked on the first dimension.
        timestep: The time step for the Runge-Kutta method.
        hamiltonian: The Hamiltonian in matrix form.
        lindbladian: The Lindbladian jump operators in matrix form, stacked on the first dimension.
    """
    k1 = eval_lindbladian(rho, hamiltonian, lindbladian)
    k2 = eval_lindbladian(rho + 0.5 * timestep * k1, hamiltonian, lindbladian)
    k3 = eval_lindbladian(rho + 0.5 * timestep * k2, hamiltonian, lindbladian)
    k4 = eval_lindbladian(rho + timestep * k3, hamiltonian, lindbladian)
    step = timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    step = 1 / 2 * (step + np.swapaxes(step.conj(), -1, -2))
    new_rho = rho + step
    return new_rho
