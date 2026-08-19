# helpers for channels
import numpy as np


def multiply_kraus_channels(kraus1: np.ndarray, kraus2: np.ndarray) -> np.ndarray:
    prod = np.einsum("ijk,mkl->imjl", kraus1, kraus2)
    shape1 = kraus1.shape
    shape2 = kraus2.shape
    return prod.reshape(shape1[0] * shape2[0], shape1[1], shape1[2])


def apply_kraus_channel(rho: np.ndarray, kraus: np.ndarray) -> np.ndarray:
    return np.einsum("ijk,...kl,iml->...jm", kraus, rho, kraus.conj())


def apply_unitary_channel(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...jk,lk->...il", unitary, rho, unitary.conj())
