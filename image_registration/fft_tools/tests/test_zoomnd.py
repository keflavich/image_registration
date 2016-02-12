from .. import zoom
import numpy as np
import pytest
import itertools

def gaussian(x):
    return np.exp(-x**2/2.)

@pytest.mark.parametrize(('imsize','ndim'),
    list(itertools.product(range(3,19),[2])))
def test_inds(imsize, ndim):
    """
    Make sure output indices are correct (don't care about zoomed values for this test)
    """
    upsample_factor = 1
    inds = np.indices([imsize]*ndim)
    rr = np.sum([(ind - (imsize-1)/2.)**2 for ind in inds],axis=0)**0.5
    gg = gaussian(rr)
    xz,zz = zoom.zoomnd(gg,usfac=upsample_factor,return_xouts=True)
    assert np.all(inds==xz)

@pytest.mark.parametrize(('imsize','upsample_factor','ndim'),
    list(itertools.product(range(3,19),range(1,25),[2])))
def test_inds(imsize, upsample_factor, ndim):
    offset = [0]*ndim
    inds = np.indices([imsize]*ndim)
    rr = np.sum([(ind - (imsize-1)/2.)**2 for ind in inds],axis=0)**0.5
    gg = gaussian(rr)
    outsize = imsize*upsample_factor
    xz,zz = zoom.zoomnd(gg,usfac=upsample_factor,outshape=[outsize]*ndim, return_xouts=True,offsets=offset)
    # wrong newinds = np.indices([imsize]*ndim)/float(imsize*upsample_factor-1)*(imsize-1)
    newinds = np.linspace(-0.5+1./upsample_factor/2.,(imsize-1)+0.5-1./upsample_factor/2.,outsize)

    for dnum in range(ndim):
        slices = [np.newaxis]*(dnum) + [slice(None)] + [np.newaxis]*(ndim-dnum-1)
        np.testing.assert_array_almost_equal(newinds[slices]*np.ones(xz[dnum].shape),xz[dnum])

@pytest.mark.parametrize(('imsize','upsample_factor'),
    list(itertools.product(range(11,27),range(1,25))))
def test_zoom_samesize(imsize, upsample_factor,doplot=False,ndim=2):
    """
    Test that zooming in by some factor with the same input & output sizes
    works
    """
    inds = np.indices([imsize]*ndim)
    rr = ((inds-(imsize-1)/2.)**2).sum(axis=0)**0.5
    gg = gaussian(rr)
    xz,zz = zoom.zoomnd(gg,usfac=upsample_factor,return_xouts=True)
    xr = ((xz - (imsize-1.)/2.)**2).sum(axis=0)**0.5

    expected_accuracy = ( (upsample_factor**1.1) * 6.2e-4 * (imsize%2==1) + # odd case
                          (upsample_factor**2) * 1.5e-4 * (imsize%2==0)   + # even case
                          0.002 ) # constant offset because the above is a fit
                          

    assert ((gaussian(xr)-zz)**2).sum() < expected_accuracy


@pytest.mark.parametrize(('imsize','upsample_factor'),
    list(itertools.product(range(11,27),range(1,25))))
def test_zoom_fullsize(imsize, upsample_factor,doplot=False,ndim=2):
    """
    Test that zooming in by some factor with output size = input size * upsample factor
    """
    inds = np.indices([imsize]*ndim)
    rr = ((inds-(imsize-1)/2.)**2).sum(axis=0)**0.5
    gg = gaussian(rr)
    outshape = [s*upsample_factor for s in gg.shape]
    xz,zz = zoom.zoomnd(gg,usfac=upsample_factor,outshape=outshape,return_xouts=True)
    xr = ((xz - (imsize-1.)/2.)**2).sum(axis=0)**0.5

    expected_accuracy = ( (upsample_factor**2) * 1.2e-4 + # odd case
                          (upsample_factor**2) * 4.9e-4 * (imsize%2==0)   + # even case
                          0.002 ) # constant offset because the above is a fit
                          

    assert ((gaussian(xr)-zz)**2).sum() < expected_accuracy


@pytest.mark.parametrize(('imsize','upsample_factor','offset'),
    list(itertools.product(range(11,27),range(1,25),((-1,1),(-1.5,3)))))
def test_zoom_samesize_uncentered(imsize, upsample_factor, offset, doplot=False,ndim=2):
    """
    Test that zooming in by some factor with the same input & output sizes

    allow for non-centered input images
    """
    inds = np.indices([imsize]*ndim)
    rr = ((inds[0] - (imsize-1)/2. - offset[0])**2  + (inds[1] - (imsize-1)/2. - offset[1])**2)**0.5
    gg = gaussian(rr)
    xz,zz = zoom.zoomnd(gg,usfac=upsample_factor,return_xouts=True)
    xr = ((xz[0] - (imsize-1)/2. - offset[0])**2  + (xz[1] - (imsize-1)/2. - offset[1])**2)**0.5

    expected_accuracy = ( (upsample_factor**1.1) * 6.2e-4 * (imsize%2==1) + # odd case
                          (upsample_factor**2) * 1.5e-4 * (imsize%2==0)   + # even case
                          0.002 ) # constant offset because the above is a fit
                          

    assert ((gaussian(xr)-zz)**2).sum() < expected_accuracy

@pytest.mark.parametrize(('imsize','upsample_factor','offset'),
    list(itertools.product(range(11,27),range(1,25),((-1,1),(-1.5,3)))))
def test_zoom_samesize_recentered(imsize, upsample_factor, offset, doplot=False,ndim=2):
    """
    Test that zooming in by some factor with the same input & output sizes

    allow for non-centered input images, AND zoom back in on the center
    """
    inds = np.indices([imsize]*ndim)
    rr = ((inds[0] - (imsize-1)/2. - offset[0])**2  + (inds[1] - (imsize-1)/2. - offset[1])**2)**0.5
    gg = gaussian(rr)
    xz,zz = zoom.zoomnd(gg,usfac=upsample_factor,return_xouts=True,offsets=offset)
    xr = ((xz[0] - (imsize-1)/2. - offset[0])**2  + (xz[1] - (imsize-1)/2. - offset[1])**2)**0.5

    expected_accuracy = ( (upsample_factor**1.1) * 6.2e-4 * (imsize%2==1) + # odd case
                          (upsample_factor**2) * 1.5e-4 * (imsize%2==0)   + # even case
                          0.002 ) # constant offset because the above is a fit
                          

    assert ((gaussian(xr)-zz)**2).sum() < expected_accuracy


