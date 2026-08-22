# tests for the paulis

import numpy as np
import pytest

from quantumsim.basis.pauli import get_pauli_product


@pytest.mark.parametrize(
    "paulistr",
    [
        (1,),
        (1, 3),
        (1, 2, 2),
        (1, 3, 0, 2),
    ],
)
def test_size_and_unitarity(paulistr):
    pauli = get_pauli_product(paulistr)
    assert pauli.shape == (2 ** len(paulistr), 2 ** len(paulistr))
    assert np.allclose(pauli @ pauli.conj().T, np.eye(2 ** len(paulistr)))
