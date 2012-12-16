from image_registration.fft_tools import upsample
import numpy as np
import pytest
import itertools


def gaussian(imsize,cx=0,cy=0):
    xx,yy = np.meshgrid(np.linspace(-5,5,imsize),np.linspace(-5,5,imsize))
    data = np.exp(-((xx-cx)**2+(yy-cy)**2)/(0.5**2 * 2.))
    return data

def zoomed_gaussian(imsize,cx=0,cy=0,upsample_factor=1.):
    grid = np.linspace(-5/float(upsample_factor),5/float(upsample_factor),imsize)
    xx,yy = np.meshgrid(grid,grid)
    data = np.exp(-((xx-cx)**2+(yy-cy)**2)/(0.5**2 * 2.))
    return data

upsample_factors = (7,8) # (5,3,2,1)
iterpars_odd = (
    list(itertools.product((99,101),(99,100),(0,),(0,),upsample_factors))+
    list(itertools.product((99,101),(99,100),(4,-5),(6,-3),upsample_factors))
    )

iterpars_even = (
    list(itertools.product((100,102),(100,101),(0,),(0,),upsample_factors))+
    list(itertools.product((100,102),(100,101),(4,-5),(6,-3),upsample_factors))
    )

from pylab import * # debug

@pytest.mark.parametrize(('imsize','outsize','cx','cy','upsample_factor'),iterpars_odd)
def test_center_zoom_odd(imsize,outsize,cx,cy,upsample_factor):
    image = gaussian(imsize)
    pixspace = np.mean(np.diff(np.linspace(-5,5,imsize)))
    image_shifted = gaussian(imsize,cx*pixspace,cy*pixspace)
    zoom_pixspace = np.mean(np.diff(np.linspace(-5/float(upsample_factor),5/float(upsample_factor),imsize)))
    image_shifted_zoomed = zoomed_gaussian(imsize,cx*zoom_pixspace,cy*zoom_pixspace,upsample_factor)
    dmax = np.unravel_index(image_shifted_zoomed.argmax(),image.shape)
    x,y,zoom = upsample.center_zoom_image(image, upsample_factor=upsample_factor, output_size=outsize, nthreads=4,
            xshift=cx, yshift=cy, return_axes=True)


    zmax = np.unravel_index(zoom.argmax(),zoom.shape)
    #print 'image position of max:',dmax,' zoom position of max:',zmax
    #print 'x,y max: ',x[zmax],y[zmax]
    clf()
    if upsample_factor >= 2 and False:
        crop1 = zoom[::upsample_factor,::upsample_factor]
        crop2slice = slice(imsize/2.-imsize/upsample_factor/2.,imsize/2.+imsize/upsample_factor/2.)
        crop2 = image_shifted_zoomed[crop2slice,crop2slice]
        if crop1.shape == crop2.shape:
            imshow(crop1-crop2)
    elif image.shape == zoom.shape:
        figure(1)
        clf()
        imshow(zoom-image_shifted_zoomed)
        colorbar()
        title("%i,%i,%i,%i,%i" % (imsize,outsize,cx,cy,upsample_factor))
        figure(2)
        clf()
        subplot(121)
        imshow(zoom)
        colorbar()
        subplot(122)
        imshow(image_shifted_zoomed)
        colorbar()
        title("%i,%i,%i,%i,%i" % (imsize,outsize,cx,cy,upsample_factor))
        figure(3)
        clf()
        imshow(image_shifted)
        contour(x,y,zoom,cmap=cm.gray)
        title("%i,%i,%i,%i,%i" % (imsize,outsize,cx,cy,upsample_factor))
    vshape = image.shape[0]*upsample_factor,image.shape[1]*upsample_factor
    s1,s2 = outsize,outsize
    roff = -int(np.round(float(vshape[0] - upsample_factor - s1)/2)) - (upsample_factor%2==0)
    coff = -int(np.round(float(vshape[1] - upsample_factor - s2)/2)) - (upsample_factor%2==0)


    print " ".join(["%8.4f" % q for q in 
        (upsample_factor,imsize,outsize,cx,cy,x[zmax],y[zmax],dmax[1],dmax[0],x[zmax]-dmax[1],y[zmax]-dmax[0],
            cx/float(upsample_factor),cy/float(upsample_factor))])
    # dmax can never be non-int
    assert (x[zmax]) == dmax[1]
    assert (y[zmax]) == dmax[0]
    #if upsample_factor == 1:
    #    assert dmax[0] == zmax[0]
    #    assert dmax[1] == zmax[1]
    if image.shape==zoom.shape:
        assert ((zoom-image_shifted_zoomed)**2).sum() < 0.001


@pytest.mark.parametrize(('imsize','outsize','cx','cy','upsample_factor'),iterpars_even)
def test_center_zoom_even(imsize,outsize,cx,cy,upsample_factor):
    pixspace = np.mean(np.diff(np.linspace(-5,5,imsize)))
    image = gaussian(imsize,cx=pixspace/2,cy=pixspace/2)
    zoom_pixspace = np.mean(np.diff(np.linspace(-5/float(upsample_factor),5/float(upsample_factor),imsize)))
    image_shifted = zoomed_gaussian(imsize,cx*zoom_pixspace+zoom_pixspace/2,cy*zoom_pixspace+zoom_pixspace/2,upsample_factor)
    dmax = np.unravel_index(image_shifted.argmax(),image.shape)
    x,y,zoom = upsample.center_zoom_image(image, upsample_factor=upsample_factor, output_size=outsize, nthreads=4,
            xshift=cx, yshift=cy, return_axes=True)

    zmax = np.unravel_index(zoom.argmax(),zoom.shape)
    print 'image position of max:',dmax,' zoom position of max:',zmax
    print 'x,y max: ',x[zmax],y[zmax]
    clf()
    if upsample_factor == 2:
        crop1 = zoom[::upsample_factor,::upsample_factor]
        crop2 = image_shifted[imsize/4.:imsize*3/4.,imsize/4.:imsize*3/4.]
        if crop1.shape == crop2.shape:
            imshow(crop1-crop2)
    elif image.shape == zoom.shape:
        figure(1)
        clf()
        imshow(zoom-image_shifted)
        colorbar()
        figure(2)
        clf()
        subplot(121)
        imshow(zoom)
        subplot(122)
        imshow(image_shifted)
    vshape = image.shape[0]*upsample_factor,image.shape[1]*upsample_factor
    s1,s2 = outsize,outsize
    roff = -int(float(vshape[0] - upsample_factor - s1)/2) 
    coff = -int(float(vshape[1] - upsample_factor - s2)/2) 
    assert round(x[zmax]) == dmax[1]
    assert round(y[zmax]) == dmax[0]
    if upsample_factor == 1 and image.shape==zoom.shape:
        assert dmax[0] == zmax[0]
        assert dmax[1] == zmax[1]
    if image.shape==zoom.shape:
        assert ((zoom-image_shifted)**2).sum() < 0.001


