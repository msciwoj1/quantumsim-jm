# utils for random matrix generation
import numpy as np


def ginibre_matrix(n: int, rng: np.random.Generator | None = None):
    """Return a complex Ginibre matrix of size n x n."""
    rng = rng or np.random.default_rng()
    return rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))


def haar_matrix(n: int, rng: np.random.Generator | None = None):
    """Return a Haar-distributed unitary matrix of size n x n.

    Uses the method by Mezzadri (arxiv.org/abs/math-ph/0609050v2).
    """
    rng = rng or np.random.default_rng()
    complex_matrix = ginibre_matrix(n, rng)
    ortho, upper = np.linalg.qr(complex_matrix)
    norm = np.diag(np.sign(np.diagonal(upper)))
    return ortho @ norm


def hilbert_schmidt_matrix(n: int, rng: np.random.Generator | None = None):
    """Return a Hilbert–Schmidt-distributed matrix of size n x n describing a mixed quantum state.

    Uses the method by Zyczkowski et al (arxiv.org/abs/1010.3570v2).
    """
    rng = rng or np.random.default_rng()
    complex_matrix = ginibre_matrix(n, rng)
    prod = complex_matrix @ complex_matrix.conj().T
    return prod / np.trace(prod)


def bures_matrix(n: int, rng: np.random.Generator | None = None):
    """Return a Bures-distributed matrix of size n x n describing a mixed quantum state.

    Uses the method by Zyczkowski et al (arxiv.org/abs/1010.3570v2).
    """
    rng = rng or np.random.default_rng()
    complex_matrix = ginibre_matrix(n, rng)
    unitary_matrix = haar_matrix(n, rng)
    prod = (
        (np.eye(n) + unitary_matrix)
        @ complex_matrix
        @ complex_matrix.conj().T
        @ (np.eye(n) + unitary_matrix.conj().T)
    )
    return prod / np.trace(prod)
