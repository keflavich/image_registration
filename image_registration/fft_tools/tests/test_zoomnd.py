from image_registration.fft_tools import upsample
import numpy as np
import pytest
import itertools

def gaussian(x):
    return np.exp(-x**2/2.)

@pytest.mark.parametrize(('imsize','upsample_factor'),
    list(itertools.product(range(11,27),range(1,25))))
def test_plot_tests(imsize, upsample_factor,doplot=False,ndim=2):
    inds = np.indices([imsize]*ndim)
    rr = ((inds-(imsize-1)/2.)**2).sum(axis=0)**0.5


