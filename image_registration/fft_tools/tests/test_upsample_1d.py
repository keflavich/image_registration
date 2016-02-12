from ..fft_tools import zoom
import numpy as np
import pytest
import itertools

def gaussian(x):
    return np.exp(-x**2/2.)

@pytest.mark.parametrize(('imsize','upsample_factor'),
    list(itertools.product(range(20,27),range(1,25))))
def test_plot_tests(imsize, upsample_factor,doplot=False):
    """
    Test that, when upsampled, z[::upsample_factor] = g

    Tested for offsets in the *input gaussian function*

    This doesn't imply you know what's going on, just that the behavior is consistent

    ONLY tests for OUTSIZE=INSIZE*UPSAMPLE_FACTOR
    """
    # always centered
    x = np.arange(imsize)
    g = gaussian(x - (imsize-1)/2.)
    xz,z = zoom.zoom1d(g,upsample_factor,outsize=g.size*upsample_factor,return_xouts=True)

    if doplot:
        print(x,xz)
        import pylab as pl
        pl.figure(1).clf()
        pl.plot(xz,z)
        pl.plot(x,g,label='input')
        pl.plot(xz,gaussian(xz-(imsize-1)/2.))
        pl.figure(2).clf()
        pl.plot(z[::upsample_factor])
        pl.plot(g)
        pl.figure(3).clf()
        pl.plot(gaussian(xz-(imsize-1)/2.)-z)
    assert ((gaussian(xz - (imsize-1)/2.)-z)**2).sum() < 4e-5*upsample_factor+0.12*imsize**-1.83

@pytest.mark.parametrize(('imsize','upsample_factor','offset'),
    list(itertools.product(range(20,27),range(1,25),(-1,0,1))))
def test_gaussian_upsample(imsize, upsample_factor, offset):
    """
    Test that, when upsampled, z[::upsample_factor] = g

    Tested for offsets in the *input gaussian function*

    This doesn't imply you know what's going on, just that the behavior is consistent

    ONLY tests for OUTSIZE=INSIZE*UPSAMPLE_FACTOR
    """
    x = np.arange(imsize)
    g = gaussian(x-(imsize-1)/2.-offset)
    xz,z = zoom.zoom1d(g,upsample_factor,outsize=g.size*upsample_factor,return_xouts=True)

    assert ((gaussian(xz - (imsize-1)/2. -offset)-z)**2).sum() < 4e-5*upsample_factor+0.12*imsize**-1.83

@pytest.mark.parametrize(('imsize','upsample_factor','offset'),
    list(itertools.product(range(11,27),range(1,25),(-1.5,-0.5,0.5,1.5))))
def test_gaussian_upsample_oddoffsets(imsize, upsample_factor, offset):
    """
    Test that, when upsampled, z[::upsample_factor] = g

    Tested for offsets in the *input gaussian function*

    This doesn't imply you know what's going on, just that the behavior is consistent

    ONLY tests for OUTSIZE=INSIZE*UPSAMPLE_FACTOR
    """
    x = np.arange(imsize)
    g = gaussian(x-(imsize-1)/2.-offset)
    xz,z = zoom.zoom1d(g,upsample_factor,outsize=g.size*upsample_factor,return_xouts=True)

    assert ((gaussian(xz - (imsize-1)/2. -offset)-z)**2).sum() < 4e-5*upsample_factor+0.12*imsize**-1.83
