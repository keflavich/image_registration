from image_registration.fft_tools import upsample
import numpy as np
import pytest
import itertools

def gaussian(imsize, offset=0):
    x = np.linspace(-5,5,imsize)
    g = np.exp(-(x-offset)**2)
    return g

def gaussian2(imsize, upsample_factor=1):
    inds = np.arange(imsize,dtype='float')
    x = inds/(imsize-1) - 0.5 
    g = np.exp(-(x*5/upsample_factor)**2)
    return g

@pytest.mark.parametrize(('imsize','upsample_factor','offset'),
    list(itertools.product(range(5,27),range(1,25),(-1,0,1))))
def test_gaussian_upsample(imsize, upsample_factor, offset):
    """
    Test that, when upsampled, z[::upsample_factor] = g

    Tested for offsets in the *input gaussian function*

    This doesn't imply you know what's going on, just that the behavior is consistent

    ONLY tests for OUTSIZE=INSIZE*UPSAMPLE_FACTOR
    """
    g = gaussian(imsize,offset)
    z = upsample.zoom1d(g,upsample_factor,outsize=g.size*upsample_factor)
    assert ((g-z[::upsample_factor])**2).sum() < 0.001

@pytest.mark.parametrize(('imsize','upsample_factor','offset'),
    list(itertools.product(range(5,27),range(1,25),(-1,0,1))))
def test_gaussian_centered(imsize, upsample_factor, offset):
    """
    Test to make sure that putting in an offset into the zoom function
    appropriately moves the zoomed peak to the center of the image


    """
    g = gaussian(imsize,offset)
    gcentered = gaussian(imsize)
    z = upsample.zoom1d(g,upsample_factor,offset=offset,outsize=g.size*upsample_factor)
    assert ((gcentered-z[::upsample_factor])**2).sum() < 0.001
