from image_registration.fft_tools import zoom
import numpy as np
import matplotlib.pyplot as pl

def iterative_zoom(image, mindiff=1., zoomshape=[10,10],
        return_zoomed=False, zoomstep=2, verbose=False,
        minmax=np.min, ploteach=False, return_center=True):
    """
    Iteratively zoom in on the *minimum* position in an image until the
    delta-peak value is below `mindiff`

    Parameters
    ----------
    image : np.ndarray
        Two-dimensional image with a *minimum* to zoom in on (or maximum, if
        specified using `minmax`)
    mindiff : float
        Minimum difference that must be present in image before zooming is done
    zoomshape : [int,int]
        Shape of the "mini" image to create.  Smaller is faster, but a bit less
        accurate.  [10,10] seems to work well in preliminary tests (though unit
        tests have not been written)
    return_zoomed : bool
        Return the zoomed image in addition to the measured offset?
    zoomstep : int
        Amount to increase the zoom factor by on each iteration.  Probably best to
        stick with small integers (2-5ish).
    verbose : bool
        Print out information about zoom factor, offset at each iteration
    minmax : np.min or np.max
        Can zoom in on the minimum or maximum of the image
    ploteach : bool
        Primarily a debug tool, and to be used with extreme caution!  Will open
        a new figure at each iteration showing the next zoom level.
    return_center : bool
        Return the center position in original image coordinates?  If False,
        will retern the *offset from center* instead (but beware the
        conventions associated with the concept of 'center' for even images).

    Returns
    -------
    The y,x offsets (following numpy convention) of the center position of the
    original image.  If `return_zoomed`, returns (zoomed_image, zoom_factor,
    offsets) because you can't interpret the zoomed image without the zoom
    factor.
    """

    image_zoom = image

    argminmax = np.argmin if "min" in minmax.__name__ else np.argmax

    zf = 1. # "zoom factor" initialized to 1 for the base shift measurement
    offset = np.array([0]*image.ndim,dtype='float') # center offset
    delta_image = (image_zoom - minmax(image_zoom))
    xaxzoom = np.indices(image.shape)

    if ploteach:
        ii = 1
        pl.figure(ii)
        pl.clf()
        pl.pcolor(np.arange(image.shape[0]+1)-0.5,np.arange(image.shape[1]+1)-0.5, image)
        minpos = np.unravel_index(argminmax(image_zoom), image_zoom.shape)
        pl.plot(minpos[1],minpos[0],'wx')

    # check to make sure the smallest *nonzero* difference > mindiff
    while np.abs(delta_image[np.abs(delta_image)>0]).min() > mindiff:
        minpos = np.unravel_index(argminmax(image_zoom), image_zoom.shape)
        center = xaxzoom[0][minpos],xaxzoom[1][minpos]
        offset = xaxzoom[0][minpos]-(image.shape[0]-1)/2,xaxzoom[1][minpos]-(image.shape[1]-1)/2

        zf *= zoomstep

        xaxzoom, image_zoom = zoom.zoom_on_pixel(image, center, usfac=zf,
                outshape=zoomshape, return_xouts=True)
        delta_image = image_zoom-minmax(image_zoom)

        # base case: in case you can't do any better...
        # (at this point, you're all the way zoomed)
        if np.all(delta_image == 0):
            if verbose:
                print "Can't zoom any further.  zf=%i" % zf
            break

        if verbose:
            print ("Zoom factor %6i, center = %30s, offset=%30s, minpos=%30s" %
                    (zf, ",".join(["%15g" % c for c in center]),
                         ",".join(["%15g" % c for c in offset]),
                         ",".join(["%15g" % c for c in minpos]),
                         ))
        if ploteach:
            ii += 1
            pl.figure(ii)
            pl.clf()
            pl.pcolor(centers_to_edges(xaxzoom[1][0,:]),centers_to_edges(xaxzoom[0][:,0]),image_zoom)
            pl.contour(xaxzoom[1],xaxzoom[0],image_zoom-image_zoom.min(),levels=[1,5,15],cmap=pl.cm.gray)
            pl.plot(center[1],center[0],'wx')
            minpos = np.unravel_index(argminmax(image_zoom), image_zoom.shape)
            pl.plot(xaxzoom[1][minpos],
                    xaxzoom[0][minpos],
                    'w+')
            pl.arrow(center[1],center[0],xaxzoom[1][minpos]-center[1],xaxzoom[0][minpos]-center[0],color='w',
                    head_width=0.1/zf, linewidth=1./zf, length_includes_head=True)
            pl.figure(1)
            #pl.contour(xaxzoom[1],xaxzoom[0],image_zoom-image_zoom.min(),levels=[1,5,15],cmap=pl.cm.gray)
            pl.arrow(center[1],center[0],xaxzoom[1][minpos]-center[1],xaxzoom[0][minpos]-center[0],color='w',
                    head_width=0.1/zf, linewidth=1./zf, length_includes_head=True)
                    
    if return_center:
        result = center
    else:
        result = offset 

    if return_zoomed:
        return image_zoom,zf,result
    else:
        return result

def centers_to_edges(arr):
    dx = arr[1]-arr[0]
    newarr = np.linspace(arr.min()-dx/2,arr.max()+dx/2,arr.size+1)
    return newarr


def iterative_zoom_1d(data, mindiff=1., zoomshape=(10,),
        return_zoomed=False, zoomstep=2, verbose=False,
        minmax=np.min, return_center=True):
    """
    Iteratively zoom in on the *minimum* position in a spectrum or timestream
    until the delta-peak value is below `mindiff`

    Parameters
    ----------
    data : np.ndarray
        One-dimensional array with a *minimum* (or maximum, as specified by
        minmax) to zoom in on
    mindiff : float
        Minimum difference that must be present in image before zooming is done
    zoomshape : int
        Shape of the "mini" image to create.  Smaller is faster, but a bit less
        accurate.  10 seems to work well in preliminary tests (though unit
        tests have not been written)
    return_zoomed : bool
        Return the zoomed image in addition to the measured offset?
    zoomstep : int
        Amount to increase the zoom factor by on each iteration.  Probably best to
        stick with small integers (2-5ish).
    verbose : bool
        Print out information about zoom factor, offset at each iteration
    minmax : np.min or np.max
        Can zoom in on the minimum or maximum of the image
    return_center : bool
        Return the center position in original image coordinates?  If False,
        will retern the *offset from center* instead (but beware the
        conventions associated with the concept of 'center' for even images).

    Returns
    -------
    The x offsets of the center position of the original spectrum.  If
    `return_zoomed`, returns (zoomed_image, zoom_factor, offsets) because you
    can't interpret the zoomed spectrum without the zoom factor.
    """

    data_zoom = data

    argminmax = np.argmin if "min" in minmax.__name__ else np.argmax

    zf = 1. # "zoom factor" initialized to 1 for the base shift measurement
    offset = 0.
    delta_data = (data_zoom - minmax(data_zoom))
    xaxzoom = np.arange(data.size)

    # check to make sure the smallest *nonzero* difference > mindiff
    while np.abs(delta_data[np.abs(delta_data)>0]).min() > mindiff:
        minpos = argminmax(data_zoom)
        center = xaxzoom.squeeze()[minpos],
        offset = xaxzoom.squeeze()[minpos]-(data.size-1)/2,

        zf *= zoomstep

        xaxzoom, data_zoom = zoom.zoom_on_pixel(data, center, usfac=zf,
                outshape=zoomshape, return_xouts=True)
        delta_data = data_zoom-minmax(data_zoom)

        # base case: in case you can't do any better...
        # (at this point, you're all the way zoomed)
        if np.all(delta_data == 0):
            if verbose:
                print "Can't zoom any further.  zf=%i" % zf
            break

        if verbose:
            print ("Zoom factor %6i, center = %30s, offset=%30s, minpos=%30s, mindiff=%30s" %
                    (zf, "%15g" % center,
                         "%15g" % offset,
                         "%15g" % minpos,
                         "%15g" % np.abs(delta_data[np.abs(delta_data)>0]).min(),
                         ))
                    
    if return_center:
        result = center
    else:
        result = offset 

    if return_zoomed:
        return data_zoom,zf,result
    else:
        return result

def centers_to_edges(arr):
    dx = arr[1]-arr[0]
    newarr = np.linspace(arr.min()-dx/2,arr.max()+dx/2,arr.size+1)
    return newarr

