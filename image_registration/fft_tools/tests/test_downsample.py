from ..downsample import downsample_axis,downsample,downsample_cube,downsample_1d
from numpy.testing import assert_allclose
import numpy as np

def test_downsample_nodownsample():
    x = np.outer(np.arange(10),np.ones(10))
    xds = downsample_axis(x, 1, 0)
    assert_allclose(x,xds)

def test_downsample_matched():
    x = np.outer(np.arange(10),np.ones(10))
    xds = downsample_axis(x, 2, 0)
    answer = np.outer(np.arange(0,10,2)+0.5,np.ones(10))
    assert_allclose(answer,xds)

def test_downsample_padded():
    x = np.outer(np.arange(10),np.ones(10))
    xds = downsample_axis(x, 3, 0)
    answer = np.outer(np.array([1,4,7,9]),np.ones(10))
    assert_allclose(answer,xds)
