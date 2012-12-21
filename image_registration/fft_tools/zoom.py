import fast_ffts
import numpy as np
import scale

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

def zoomnd(inp, usfac=1, outshape=None, offsets=(), nthreads=1,
        use_numpy_fft=False, return_real=True, return_xouts=False):
    """
    Zoom in to the center of a 1D or 2D array using Fourier upsampling
    (in principle, should work on N-dimensions, but does not at present!)

    Parameters
    ----------
    inp : np.ndarray
        Input 1D array
    usfac : int
        Upsampling factor
    outshape : int
        Number of pixels in output array
    offsets : tuple of floats
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
    The input array upsampled by a factor `usfac` with size `outshape`.
    If `return_xouts`, returns a tuple (xvals, zoomed)
    """

    inshape = inp.shape
    if outshape is None: outshape=inshape

    if len(offsets) > 0 and len(offsets) != inp.ndim:
        raise ValueError("Must have same # of offsets as input dimensions")
    elif len(offsets) == 0:
        offsets = (0,) * inp.ndim

    outarr = np.zeros((inp.ndim,)+tuple(outshape),dtype='float')
    for ii,(insize, outsize, off) in enumerate(zip(inshape,outshape,offsets)):
        # output array should cover 1/usfac *  the range of the input
        # it should go from 1/2.-1/usfac to 1/2+1/usfac
        # plus whatever offset is specified
        # outsize is always 1+(highest index of input)
        middle = (insize-1)/2. + off
        outarr_d = np.linspace(middle - (outsize-1.)/usfac/2., middle + (outsize-1.)/usfac/2., outsize)
        
        # slice(None) = ":" or "get everything"
        # [None] = newaxis = add a blank axis on this dim
        dims = [None]*ii + [slice(None)] + [None]*(inp.ndim-1-ii)
        outarr[ii] = outarr_d[dims]

    # temporary hack
    if inp.ndim != 2: 
        raise NotImplementedError("Can't do more than 2D yet")
    interpfunc = scale.fourier_interp2d if inp.ndim == 2 else scale.fourier_interp1d

    result = interpfunc(inp, outarr, nthreads=nthreads,
            use_numpy_fft=use_numpy_fft, return_real=return_real)

    if return_xouts:
        return outarr,result
    else:
        return result

