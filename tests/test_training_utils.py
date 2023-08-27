import numpy as np
from cal_ratio_trainer.training.training_utils import _pad_arrays
import pytest


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_pad_arrays_same(ndim):
    "Test _pad_arrays, where template and the data have the same length and shape."
    # Create num `ndim` numpy array of length n for the template
    template_len = 5
    template = np.random.rand(*[template_len for _ in range(ndim)])

    # Create num `ndim` numpy array of length n for the data, same length.
    data_len = template_len
    data = np.random.rand(*[data_len for _ in range(ndim)])

    # Now run the pad
    r = _pad_arrays([data], [template])
    assert len(r) == 1
    assert r[0].shape == template.shape  # type: ignore


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_pad_arrays_short(ndim):
    "Test _pad_arrays, where template is longer than the data."
    # Create num `ndim` numpy array of length n for the template
    template_len = 5
    template = np.random.rand(*[template_len for _ in range(ndim)])

    # Create num `ndim` numpy array of length n for the data, same length.
    data_len = 3
    data = np.random.rand(*[data_len for _ in range(ndim)])

    # Now run the pad
    r = _pad_arrays([data], [template])
    assert len(r) == 1
    assert r[0].shape == template.shape  # type: ignore


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_and_arrays_long(ndim):
    "Test _pad_arrays, where template is shorter than the data."
    # Create num `ndim` numpy array of length n for the template
    template_len = 5
    template = np.random.rand(*[template_len for _ in range(1)])

    # Create num `ndim` numpy array of length n for the data, same length.
    data_len = 10
    data = np.random.rand(*[data_len for _ in range(1)])

    # Now run the pad
    r = _pad_arrays([data], [template])
    assert len(r) == 1
    assert r[0].shape == template.shape  # type: ignore
