import fast_ffts
import numpy as np
import scale
from matplotlib import docstring

def zoom1d(inp, usfac=1, outsize=None, offset=0, nthreads=1,
        use_numpy_fft=False, return_xouts=False, return_real=True):
    """
    Zoom in to the center of a 1D array using Fourier upsampling

    Parameters
    ----------
    inp : np.ndarray
        Input 1D array
    usfac : int
        Upsampling factor
    outsize : int
        Number of pixels in output array
    offset : float
        Offset from center *in original pixel units*

    Other Parameters
    ----------------
    return_xouts : bool
        Return the X indices of the output array in addition to the scaled
        array
    return_real : bool
        Return the real part of the zoomed array (if True) or the complex

    Returns
    -------
    The input array upsampled by a factor `usfac` with size `outsize`.
    If `return_xouts`, returns a tuple (xvals, zoomed)
    """

    insize, = inp.shape
    if outsize is None: outsize=insize

    # output array should cover 1/usfac *  the range of the input
    # it should go from 1/2.-1/usfac to 1/2+1/usfac
    # plus whatever offset is specified
    # outsize is always 1+(highest index of input)
    middle = (insize-1.)/2. + offset
    outarr = np.linspace(middle - (outsize-1)/usfac/2., middle + (outsize-1)/usfac/2., outsize)

    result = scale.fourier_interp1d(inp, outarr, nthreads=nthreads,
            use_numpy_fft=use_numpy_fft, return_real=return_real)

    if return_xouts:
        return outarr,result
    else:
        return result

def zoom_on_pixel(inp, coordinates, usfac=1, outshape=None, nthreads=1,
        use_numpy_fft=False, return_real=True, return_xouts=False):
    """
    Zoom in on a 1D or 2D array using Fourier upsampling
    (in principle, should work on N-dimensions, but does not at present!)

    Parameters
    ----------
    inp : np.ndarray
        Input 1D array
    coordinates : tuple of floats
        Pixel to zoom in on
    usfac : int
        Upsampling factor
    outshape : int
        Number of pixels in output array

    Other Parameters
    ----------------
    return_xouts : bool
        Return the X indices of the output array in addition to the scaled
        array
    return_real : bool
        Return the real part of the zoomed array (if True) or the complex

    Returns
    -------
    The input array upsampled by a factor `usfac` with size `outshape`.
    If `return_xouts`, returns a tuple (xvals, zoomed)
    """

    inshape = inp.shape
    if outshape is None: outshape=inshape

    outarr = np.zeros((inp.ndim,)+tuple(outshape),dtype='float')
    for ii,(insize, outsize, target) in enumerate(zip(inshape,outshape,coordinates)):
        # output array should cover 1/usfac *  the range of the input
        # it should go from 1/2.-1/usfac to 1/2+1/usfac
        # plus whatever offset is specified
        # outsize is always 1+(highest index of input)
        outarr_d = np.linspace(target - (outsize-1.)/usfac/2.,
                               target + (outsize-1.)/usfac/2.,
                               outsize)
        
        # slice(None) = ":" or "get everything"
        # [None] = newaxis = add a blank axis on this dim
        dims = [None]*ii + [slice(None)] + [None]*(inp.ndim-1-ii)
        outarr[ii] = outarr_d[dims]

    # temporary hack
    if inp.ndim == 1:
        result = scale.fourier_interp1d(inp, outarr.squeeze(), nthreads=nthreads,
                use_numpy_fft=use_numpy_fft, return_real=return_real)
    elif inp.ndim == 2:
        result = scale.fourier_interp2d(inp, outarr, nthreads=nthreads,
                                        use_numpy_fft=use_numpy_fft,
                                        return_real=return_real)
    else:
        raise NotImplementedError("Can't do more than 2D yet")

    if return_xouts:
        return outarr,result
    else:
        return result

def zoomnd(inp, offsets=(), middle_convention=np.float, **kwargs):
    """
    Zoom in to the center of a 1D or 2D array using Fourier upsampling
    (in principle, should work on N-dimensions, but does not at present!)

    Parameters
    ----------
    inp : np.ndarray
        Input 1D array
    offsets : tuple of floats
        Offset from center *in original pixel units*"
    middle_convention : function
        What convention to use for the "Middle" of the array.  Should be either
        float (i.e., can be half-pixel), floor, or ceil.  I don't think round makes
        a ton of sense... should just be ceil.
    usfac : int
        Upsampling factor
        (passed to :func:`zoom_on_pixel`)
    outshape : int
        Number of pixels in output array
        (passed to :func:`zoom_on_pixel`)
    
    Other Parameters
    ----------------
    return_xouts : bool
        Return the X indices of the output array in addition to the scaled
        array
        (passed to :func:`zoom_on_pixel`)
    return_real : bool
        Return the real part of the zoomed array (if True) or the complex
        (passed to :func:`zoom_on_pixel`)

    Returns
    -------
    The input array upsampled by a factor `usfac` with size `outshape`.
    If `return_xouts`, returns a tuple (xvals, zoomed)
    """

    if len(offsets) > 0 and len(offsets) != inp.ndim:
        raise ValueError("Must have same # of offsets as input dimensions")
    elif len(offsets) == 0:
        offsets = (0,) * inp.ndim

    # output array should cover 1/usfac *  the range of the input
    # it should go from 1/2.-1/usfac to 1/2+1/usfac
    # plus whatever offset is specified
    # outsize is always 1+(highest index of input)

    middlepix = [middle_convention((insize-1)/2.) + off
                 for insize,off in zip(inp.shape,offsets)]

    return zoom_on_pixel(inp, middlepix, **kwargs)

