import numpy as np
import types
from .downsample import downsample as downsample_2d
from .convolve_nd import convolvend as convolve

def smooth(image, kernelwidth=3, kerneltype='gaussian', trapslope=None,
        silent=True, psf_pad=True, interp_nan=False, nwidths='max',
        min_nwidths=6, return_kernel=False, normalize_kernel=np.sum,
        downsample=False, downsample_factor=None, ignore_edge_zeros=False,
        **kwargs):
    """
    Returns a smoothed image using a gaussian, boxcar, or tophat kernel

    Parameters
    ----------
    kernelwidth:
        width of kernel in pixels  (see definitions below)
    kerneltype:
        gaussian, boxcar, or tophat.  
        For a gaussian, uses a gaussian with sigma = kernelwidth (in pixels)
            out to [nwidths]-sigma
        A boxcar is a kernelwidth x kernelwidth square 
        A tophat is a flat circle with radius = kernelwidth

    psf_pad: [True]
        will pad the input image to be the image size + PSF.
        Slows things down but removes edge-wrapping effects (see convolve)
        This option should be set to false if the edges of your image are
        symmetric.
    interp_nan: [False]
        Will replace NaN points in an image with the
        smoothed average of its neighbors (you can still simply ignore NaN 
        values by setting ignore_nan=True but leaving interp_nan=False)
    silent: [True]
        turn it off to get verbose statements about kernel types
    return_kernel: [False]
        If set to true, will return the kernel as the
        second return value
    nwidths: ['max']
        number of kernel widths wide to make the kernel.  Set to 'max' to
        match the image shape, otherwise use any integer 
    min_nwidths: [6]
        minimum number of gaussian widths to make the kernel
        (the kernel will be larger than the image if the image size is <
        min_widths*kernelsize)
    normalize_kernel:
        Should the kernel preserve the map sum (i.e. kernel.sum() = 1)
        or the kernel peak (i.e. kernel.max() = 1) ?  Must be a *function* that can
        operate on a numpy array
    downsample:
        downsample after smoothing?
    downsample_factor:
        if None, default to kernelwidth
    ignore_edge_zeros: bool
        Ignore the zero-pad-created zeros.  This will effectively decrease
        the kernel area on the edges but will not re-normalize the kernel.
        This parameter may result in 'edge-brightening' effects if you're using
        a normalized kernel

    Note that the kernel is forced to be even sized on each axis to assure no
    offset when smoothing.
    """

    if (kernelwidth*min_nwidths > image.shape[0] or kernelwidth*min_nwidths > image.shape[1]):
        nwidths = min_nwidths
    if (nwidths!='max'):# and kernelwidth*nwidths < image.shape[0] and kernelwidth*nwidths < image.shape[1]):
        dimsize = np.ceil(kernelwidth*nwidths)
        dimsize += dimsize % 2
        yy,xx = np.indices([dimsize,dimsize])
        szY,szX = dimsize,dimsize
    else:
        szY,szX = image.shape
        szY += szY % 2
        szX += szX % 2
        yy,xx = np.indices([szY,szX])
    shape = (szY,szX)
    if not silent: print("Kernel size set to ",shape)

    kernel = make_kernel(shape, kernelwidth=kernelwidth, kerneltype=kerneltype,
            normalize_kernel=normalize_kernel, trapslope=trapslope)

    if not silent: print("Kernel of type %s normalized with %s has peak %g" % (kerneltype, normalize_kernel, kernel.max()))

    bad = (image != image)
    temp = image.copy() # to preserve NaN values
    # convolve does this already temp[bad] = 0

    # kwargs parsing to avoid duplicate keyword passing
    #if not kwargs.has_key('ignore_edge_zeros'): kwargs['ignore_edge_zeros']=True
    if not kwargs.has_key('interpolate_nan'): kwargs['interpolate_nan']=interp_nan

    # No need to normalize - normalization is dealt with in this code
    temp = convolve(temp,kernel,psf_pad=psf_pad, normalize_kernel=False,
            ignore_edge_zeros=ignore_edge_zeros, **kwargs)
    if interp_nan is False: temp[bad] = image[bad]

    if temp.shape != image.shape:
        raise ValueError("Output image changed size; this is completely impossible.")

    if downsample:
        if downsample_factor is None: downsample_factor = kernelwidth
        if return_kernel: return downsample_2d(temp,downsample_factor),downsample_2d(kernel,downsample_factor)
        else: return downsample_2d(temp,downsample_factor)
    else:
        if return_kernel: return temp,kernel
        else: return temp

def make_kernel(kernelshape, kernelwidth=3, kerneltype='gaussian',
        trapslope=None, normalize_kernel=np.sum, force_odd=False):
    """
    Create a smoothing kernel for use with `convolve` or `convolve_fft`

    Parameters
    ----------
    kernelshape : n-tuple
        A tuple (or list or array) defining the shape of the kernel.  The
        length of kernelshape determines the dimensionality of the resulting
        kernel

    Options
    -------
    kernelwidth : float
        Width of kernel in pixels  (see definitions under `kerneltype`)
    kerneltype : 'gaussian', 'boxcar', 'tophat', 'brickwall', 'airy', 'trapezoid'
        Defines the type of kernel to be generated.
        For a gaussian, uses a gaussian with sigma = `kernelwidth` (in pixels)
            i.e. kernel = exp(-r**2 / (2*sigma**2)) where r is the radius 
        A boxcar is a `kernelwidth` x `kernelwidth` square 
            e.g. kernel = (x < `kernelwidth`) * (y < `kernelwidth`)
        A tophat is a flat circle with radius = `kernelwidth`
            i.e. kernel = (r < `kernelwidth`)
        A 'brickwall' or 'airy' kernel is the airy function from optics.  It
            requires scipy.special for the bessel function.
            http://en.wikipedia.org/wiki/Airy_disk
        The trapezoid kernel is like a tophat but with sloped edges.  It is
            effectively a cone chopped off at the `kernelwidth` radius.
    trapslope : float
        Slope of the trapezoid kernel.  Only used if `kerneltype`=='trapezoid'
    normalize_kernel : function
        Function to use for kernel normalization 
    force_odd : boolean
        If set, forces the kernel to have odd dimensions (needed for convolve
        w/o ffts)

    Returns
    -------
    An N-dimensional float array

    """

    if force_odd:
        kernelshape = [n-1 if (n%2==0) else n for n in kernelshape]

    if normalize_kernel is True:
        normalize_kernel = np.sum

    if kerneltype == 'gaussian':
        rr = np.sum([(x-(x.max()+1)//2)**2 for x in np.indices(kernelshape)],axis=0)**0.5
        kernel = np.exp(-(rr**2)/(2.*kernelwidth**2))
        kernel /= normalize_kernel(kernel) #/ (kernelwidth**2 * (2*np.pi))

    elif kerneltype == 'boxcar':
        kernel = np.zeros(kernelshape,dtype='float64')
        kernelslices = []
        for dimsize in kernelshape:
            center = dimsize - (dimsize+1)//2
            kernelslices += [slice(center - (kernelwidth)//2, center + (kernelwidth+1)//2)]
        kernel[kernelslices] = 1.0
        kernel /= normalize_kernel(kernel)
    elif kerneltype == 'tophat':
        rr = np.sum([(x-(x.max())/2.)**2 for x in np.indices(kernelshape)],axis=0)**0.5
        kernel = np.zeros(kernelshape,dtype='float64')
        kernel[rr<kernelwidth] = 1.0
        # normalize
        kernel /= normalize_kernel(kernel)
    elif kerneltype in ('brickwall','airy'):
        try:
            import scipy.special
        except ImportError:
            raise ImportError("Could not import scipy.special; cannot create an "+
                    "airy kernel without this (need the bessel function)")
        rr = np.sum([(x-(x.max())/2.)**2 for x in np.indices(kernelshape)],axis=0)**0.5
        # airy function is first bessel(x) / x  [like the sinc]
        kernel = j1(rr/kernelwidth) / (rr/kernelwidth) 
        # fix NAN @ center
        kernel[rr==0] = 0.5
        kernel /= normalize_kernel(kernel)
    elif kerneltype == 'trapezoid':
        rr = np.sum([(x-(x.max())/2.)**2 for x in np.indices(kernelshape)],axis=0)**0.5
        if trapslope:
            zz = rr.max()-(rr*trapslope)
            zz[zz<0] = 0
            zz[rr<kernelwidth] = 1.0
            kernel = zz/zz.sum()
        else:
            raise ValueError("Must specify a slope for kerneltype='trapezoid'")

    return kernel
