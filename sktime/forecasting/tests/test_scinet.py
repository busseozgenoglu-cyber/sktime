"""Tests for SCINetForecaster."""

import numpy as np
import pytest

from sktime.forecasting.scinet import SCINetForecaster


def test_scinet_forwards_num_stacks_to_network():
    """SCINetForecaster forwards num_stacks instead of hid_size."""
    pytest.importorskip("torch")

    forecaster = SCINetForecaster(
        seq_len=4,
        hid_size=1,
        num_stacks=2,
        num_levels=1,
    )
    forecaster._y = np.zeros((4, 1))

    network = forecaster._build_network(fh=2)

    assert hasattr(network, "blocks2")
