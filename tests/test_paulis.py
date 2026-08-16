# tests for the paulis

import numpy as np
import pytest

from quantumsim.pauli import get_pauli_product


@pytest.mark.parametrize(
    "paulistr",
    [
        "X",
        "XZ",
        "XYY",
        "XZIY",
    ],
)
def test_size_and_unitarity(paulistr):
    pauli = get_pauli_product(paulistr)
    assert pauli.shape == (2 ** len(paulistr), 2 ** len(paulistr))
    assert np.allclose(pauli @ pauli.conj().T, np.eye(2 ** len(paulistr)))
