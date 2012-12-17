from image_registration.fft_tools import upsample
import numpy as np
import pytest
import itertools

def gaussian_centered(imsize, upsample_factor=1):
    """
    Create a narrow Gaussian with peak at the center of the image if
    dims are odd or center+1 if dims are even
    """
    if imsize % 2 == 1:
        xx,yy = np.meshgrid(np.linspace(-5,5,imsize),np.linspace(-5,5,imsize))
        data = np.exp(-((xx)**2+(yy)**2)/(0.5**2 * 2. * upsample_factor**2))
        return data
    else:
        xx,yy = np.meshgrid(np.linspace(-5,5,imsize+1),np.linspace(-5,5,imsize+1))
        data = np.exp(-((xx)**2+(yy)**2)/(0.5**2 * 2. * upsample_factor**2))
        return data[:-1,:-1]

def gaussian(imsize,cx=0,cy=0):
    if imsize % 2 == 1:
        xx,yy = np.meshgrid(np.linspace(-5,5,imsize),np.linspace(-5,5,imsize))
        data = np.exp(-((xx-cx)**2+(yy-cy)**2)/(0.5**2 * 2.))
        return data
    else:
        xx,yy = np.meshgrid(np.linspace(-5,5,imsize+1),np.linspace(-5,5,imsize+1))
        data = np.exp(-((xx-cx)**2+(yy-cy)**2)/(0.5**2 * 2.))
        return data[:-1,:-1]

def zoomed_gaussian(imsize,cx=0,cy=0,upsample_factor=1.):
    grid = np.linspace(-5/float(upsample_factor),5/float(upsample_factor),imsize)
    xx,yy = np.meshgrid(grid,grid)
    data = np.exp(-((xx-cx)**2+(yy-cy)**2)/(0.5**2 * 2.))
    return data

upsample_factors = (2,3,4,5,10) # (5,3,2,1)
iterpars_odd = (
    list(itertools.product((25,),(25,),(0,),(0,),upsample_factors))+
    list(itertools.product((25,),(25,),(4,-5),(6,-3),upsample_factors))
    )

iterpars_even = (
    list(itertools.product((25,26,),(25,26,),(0,),(0,),upsample_factors))+
    list(itertools.product((25,26,),(25,26,),(4,-5),(6,-3),upsample_factors))
    )

from pylab import * # debug

@pytest.mark.parametrize(('imsize','upsample_factor'),
    list(itertools.product((3,4,5,6,7,8,9,25,26,27),upsample_factors)))
def test_center_zoom_simple(imsize,upsample_factor,doplot=False):
    image = gaussian_centered(imsize)
    zoomed_image = gaussian_centered(imsize, upsample_factor)
    immax = np.unravel_index(image.argmax(),image.shape)
    zimmax = np.unravel_index(zoomed_image.argmax(),image.shape)
    assert immax[0] == floor(image.shape[0]/2.)
    assert zimmax[0] == floor(zoomed_image.shape[0]/2.)

    x,y,zoom = upsample.center_zoom_image(image,
            upsample_factor=upsample_factor, output_size=image.shape,
            nthreads=4, return_axes=True)

    if imsize * upsample_factor < 512:
        fx,fy,fullzoom = upsample.center_zoom_image(image,
                upsample_factor=upsample_factor, nthreads=4, return_axes=True)
    else:
        fullzoom = None

    if doplot:
        plotthings(image,image,zoomed_image,zoom,0,0,upsample_factor,imsize,imsize,x,y,
                fullzoom=fullzoom)

    zmax = np.where(np.abs(zoom-zoom.max()) < 1e-8)
    assert np.round(np.mean(y[zmax])) == floor(zoomed_image.shape[0]/2.)
    assert np.round(np.mean(x[zmax])) == floor(zoomed_image.shape[1]/2.)

    if len(zmax[0]) == 1:
        assert zmax[0][0] == floor(zoomed_image.shape[0]/2.)
        assert zmax[1][0] == floor(zoomed_image.shape[1]/2.)

def plotthings(image,image_shifted,image_shifted_zoomed,zoom,cx,cy,upsample_factor,imsize,outsize,x,y,
        fullzoom=None):
    figure(1)
    clf()
    imshow(zoom-image_shifted_zoomed)
    colorbar()
    title("%i,%i,%i,%i,%i" % (imsize,outsize,cx,cy,upsample_factor))
    savefig("fig1_%i_%i_%i_%i_%i.png" % (imsize,outsize,cx,cy,upsample_factor))
    figure(2)
    clf()
    subplot(121)
    imshow(zoom)
    colorbar()
    subplot(122)
    imshow(image_shifted_zoomed)
    contour(zoom,cmap=cm.gray)
    colorbar()
    title("%i,%i,%i,%i,%i" % (imsize,outsize,cx,cy,upsample_factor))
    savefig("fig2_%i_%i_%i_%i_%i.png" % (imsize,outsize,cx,cy,upsample_factor))
    fig3=figure(3)
    clf()
    imshow(image_shifted)
    contour(x,y,zoom,cmap=cm.gray)
    contour(x,y,image_shifted_zoomed,cmap=cm.spectral)
    title("%i,%i,%i,%i,%i" % (imsize,outsize,cx,cy,upsample_factor))
    savefig("fig3_%i_%i_%i_%i_%i.png" % (imsize,outsize,cx,cy,upsample_factor))
    xll = x.min()-np.mean(np.diff(x[0,:]))/2.
    yll = y.min()-np.mean(np.diff(y[:,0]))/2.
    xwidth = (x.max()-x.min()) + np.mean(np.diff(x[0,:]))
    ywidth = (y.max()-y.min()) + np.mean(np.diff(y[:,0]))
    gca().add_patch(mpl.patches.Rectangle((xll,yll),xwidth,ywidth,facecolor='none',edgecolor='w'))
    if fullzoom is not None:
        figure(4)
        clf()
        imshow(fullzoom)
        axlim = gca().axis()
        plot(fullzoom.shape[1]/2.,fullzoom.shape[0]/2.,'wx',mew=3)
        fzmax = np.unravel_index(fullzoom.argmax(),fullzoom.shape)
        plot(fzmax[1], fzmax[0],'g+',mew=3)
        title("%i,%i,%i,%i,%i" % (imsize,outsize,cx,cy,upsample_factor) + str(fzmax) + str(fullzoom.shape))
        gca().axis(axlim)
        savefig("fig4_%i_%i_%i_%i_%i.png" % (imsize,outsize,cx,cy,upsample_factor))


@pytest.mark.parametrize(('imsize','outsize','cx','cy','upsample_factor'),iterpars_even)
def test_center_zoom_even(imsize,outsize,cx,cy,upsample_factor,doplot=True):
    pixspace = np.mean(np.diff(np.linspace(-5,5,imsize)))
    image = gaussian(imsize)

    image_shifted = gaussian(imsize,cx*pixspace,cy*pixspace)
    zoom_pixspace = np.mean(np.diff(np.linspace(-5/float(upsample_factor),5/float(upsample_factor),imsize)))
    image_shifted_zoomed = zoomed_gaussian(outsize,cx*zoom_pixspace, cy*zoom_pixspace,upsample_factor)
    image_zoomed = zoomed_gaussian(outsize, 0, 0, upsample_factor)
    #image_shifted_zoomed = zoomed_gaussian(outsize,pixspace/2,pixspace/2,upsample_factor)

    middle = np.unravel_index(image.argmax(),image.shape)
    ismax = np.unravel_index(image_shifted.argmax(),image.shape)
    assert (ismax[0]-middle[0]) == cy
    assert (ismax[1]-middle[1]) == cx

    dmax = np.unravel_index(image_shifted_zoomed.argmax(),image.shape)
    x,y,zoom = upsample.center_zoom_image(image_shifted, upsample_factor=upsample_factor, output_size=outsize, nthreads=4,
            xshift=cx, yshift=cy, return_axes=True)

    zmax = np.unravel_index(zoom.argmax(),zoom.shape)
    #print 'image position of max:',dmax,' zoom position of max:',zmax
    #print 'x,y max: ',x[zmax],y[zmax]
    clf()

    vshape = image.shape[0]*upsample_factor,image.shape[1]*upsample_factor
    s1,s2 = outsize,outsize
    roff = -int(np.round(float(vshape[0] - upsample_factor - s1)/2)) - (upsample_factor%2==0)
    coff = -int(np.round(float(vshape[1] - upsample_factor - s2)/2)) - (upsample_factor%2==0)

    if doplot:
        if imsize * upsample_factor < 512:
            fx,fy,fullzoom = upsample.center_zoom_image(image, xshift=cx, yshift=cy, 
                    upsample_factor=upsample_factor, nthreads=4, return_axes=True)
        else:
            fullzoom = None
        if image.shape == zoom.shape:
            plotthings(image,image_shifted,image_zoomed,zoom,cx,cy,upsample_factor,imsize,outsize,x,y,fullzoom=fullzoom)
        else:
            plotthings(image,image_shifted,image_zoomed[:image.shape[0],:image.shape[1]],zoom[:image.shape[0],:image.shape[1]],
                    cx,cy,upsample_factor,imsize,outsize,x[:image.shape[0],:image.shape[1]],y[:image.shape[0],:image.shape[1]],fullzoom=fullzoom)

    print " ".join(["%6.2f" % q for q in 
        (upsample_factor,imsize,outsize,cx,cy,x[zmax],y[zmax],zmax[1],zmax[0],dmax[1],dmax[0],x[zmax]-dmax[1],y[zmax]-dmax[0],
            cx/float(upsample_factor),cy/float(upsample_factor),((zoom-image_zoomed)**2).sum(),
            ismax[0]-middle[0],ismax[1]-middle[1])])

    zmax = np.where(np.abs(zoom-zoom.max()) < 1e-8)
    assert np.round(np.mean(y[zmax])) == floor(image_zoomed.shape[0]/2.+cy)
    assert np.round(np.mean(x[zmax])) == floor(image_zoomed.shape[1]/2.+cx)


    #assert (x[zmax]) == dmax[1]
    #assert (y[zmax]) == dmax[0]
    #if upsample_factor == 1 and image.shape==zoom.shape:
    #    assert dmax[0] == zmax[0]
    #    assert dmax[1] == zmax[1]
    assert ((zoom-image_shifted_zoomed)**2).sum() < 0.001



"""
@pytest.mark.parametrize(('imsize','outsize','cx','cy','upsample_factor'),iterpars_odd)
def test_center_zoom_odd(imsize,outsize,cx,cy,upsample_factor):
    image = gaussian(imsize)
    pixspace = np.mean(np.diff(np.linspace(-5,5,imsize)))
    image_shifted = gaussian(imsize,cx*pixspace,cy*pixspace)
    zoom_pixspace = np.mean(np.diff(np.linspace(-5/float(upsample_factor),5/float(upsample_factor),imsize)))
    image_shifted_zoomed = zoomed_gaussian(outsize,cx*zoom_pixspace,cy*zoom_pixspace,upsample_factor)
    image_shifted_zoomed = zoomed_gaussian(outsize,0,0,upsample_factor)
    dmax = np.unravel_index(image_shifted_zoomed.argmax(),image.shape)
    ismax = np.unravel_index(image_shifted.argmax(),image.shape)
    assert ismax == (cy,cx)
    x,y,zoom = upsample.center_zoom_image(image, upsample_factor=upsample_factor, output_size=outsize, nthreads=4,
            xshift=cx, yshift=cy, return_axes=True)


    zmax = np.unravel_index(zoom.argmax(),zoom.shape)
    #print 'image position of max:',dmax,' zoom position of max:',zmax
    #print 'x,y max: ',x[zmax],y[zmax]
    clf()
    if image.shape == zoom.shape:
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
            cx/float(upsample_factor),cy/float(upsample_factor),((zoom-image_shifted_zoomed)**2).sum())])
    # dmax can never be non-int
    assert (x[zmax]) == dmax[1]
    assert (y[zmax]) == dmax[0]
    #if upsample_factor == 1:
    #    assert dmax[0] == zmax[0]
    #    assert dmax[1] == zmax[1]
    assert ((zoom-image_shifted_zoomed)**2).sum() < 0.001
"""
