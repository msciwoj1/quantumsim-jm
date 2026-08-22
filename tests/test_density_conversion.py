# Tests for density matrix conversion functions

import numpy as np
import pytest

from quantumsim.random.matrix import hilbert_schmidt_matrix
from quantumsim.transfer.basis import get_density_matrix, get_transfer_vector


@pytest.mark.parametrize(
    "dims",
    [
        (2,),
        (2, 2),
        (2, 3),
        (3, 3),
    ],
)
def test_conversion_both_ways(dims: tuple[int]):
    rng = np.random.default_rng(seed=2324)
    n = np.prod(dims)
    for _ in range(10):
        rho = hilbert_schmidt_matrix(n, rng)
        tv = get_transfer_vector(dims, rho)
        rho2 = get_density_matrix(dims, tv)
        assert np.allclose(rho, rho2)
