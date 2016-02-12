from ..fft_tools import shift
import numpy as np
import pytest
import itertools

def gaussian(x):
    return np.exp(-x**2/2.)

@pytest.mark.parametrize(('imsize','dx','dy'),
    list(itertools.product(range(9,27),np.linspace(-2.5,2.5,11),np.linspace(-2.5,2.5,11))))
def test_shifts_2d_asymm(imsize,dx,dy):

    inds = np.indices([imsize,imsize*2])
    rr = ((inds[0] - (imsize-1)/2.)**2  + (inds[1] - (imsize*2-1)/2.)**2)**0.5
    gg = gaussian(rr)

    sg = shift.shiftnd(gg,(dy,dx))

    rr2 = ((inds[0] - (imsize-1)/2. - dy)**2  + (inds[1] - (imsize*2-1)/2. - dx)**2)**0.5

    thr = gaussian(rr2)

    assert np.all(np.abs(sg-thr < 0.05))

@pytest.mark.parametrize(('imsize','dx'),
    list(itertools.product(range(9,27),np.linspace(-2.5,2.5,11))))
def test_shifts_1d(imsize,dx):
    x = np.arange(imsize)
    g = gaussian(x - (imsize-1)/2. )

    sg = shift.shiftnd(g,(dx,))

    assert np.all(np.abs(sg-gaussian(x-(imsize-1.)/2.-dx)) < 0.05)

@pytest.mark.parametrize(('imsize','dx','dy'),
    list(itertools.product(range(9,27),np.linspace(-2.5,2.5,11),np.linspace(-2.5,2.5,11))))
def test_shifts_2d(imsize,dx,dy):

    inds = np.indices([imsize]*2)
    rr = ((inds[0] - (imsize-1)/2.)**2  + (inds[1] - (imsize-1)/2.)**2)**0.5
    gg = gaussian(rr)

    sg = shift.shiftnd(gg,(dy,dx))

    rr2 = ((inds[0] - (imsize-1)/2. - dy)**2  + (inds[1] - (imsize-1)/2. - dx)**2)**0.5

    assert np.all(np.abs(sg-gaussian(rr2) < 0.05))
