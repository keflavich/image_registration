import fast_ffts
import numpy as np

def scale(data, scale_factor, out_shape=None, nthreads=1, use_numpy_fft=False,
        return_abs=False, return_real=True):
    """
    Use the fourier scale theorem to "zoom-in" an image
    http://www.cv.nrao.edu/course/astr534/FTSimilarity.html
    """
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if scale_factor <= 0:
        raise ValueError("Scale factor must be > 0")

    if out_shape is None:
        out_shape = [s*scale_factor for s in data.shape]

    inds = np.indices(data.shape)
    out_inds = np.indices(out_shape) / scale_factor

    # have to compute the DFT directly?
    #dft = 

    scaled = ifftn( fftn(data))

def scale1d(data, scale_factor, out_shape=None, nthreads=1, use_numpy_fft=False,
        return_abs=False, return_real=True):
    """
    Use the fourier scale theorem to "zoom-in" an image
    http://www.cv.nrao.edu/course/astr534/FTSimilarity.html
    """
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if scale_factor <= 0:
        raise ValueError("Scale factor must be > 0")

    if out_shape is None:
        out_shape = [s*scale_factor for s in data.shape]

    inds = np.indices(data.shape)
    freqs = fftfreq(data.shape)
    out_inds = np.indices(out_shape) / scale_factor

    # have to compute the DFT directly?
    #dft = 

    scaled = ifftn( fftn(data))

def fourier_interp1d(data, out_x, data_x=None, nthreads=1, use_numpy_fft=False,
        return_real=True):
    """
    Use the fourier scaling theorem to interpolate (or extrapolate, without raising
    any exceptions) data.

    Parameters
    ----------
    data : ndarray
        The Y-values of the array to interpolate
    out_x : ndarray
        The X-values along which the data should be interpolated
    data_x : ndarray | None
        The X-values corresponding to the data values.  If an ndarray, must
        have the same shape as data.  If not specified, will be set to
        np.arange(data.size)
    """

    # load fft
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    # specify fourier frequencies
    freq = fftfreq(data.size)[:,newaxis]

    # reshape outinds
    if out_x.ndim != 1:
        raise ValueError("Must specify a 1-d array of output indices")

    if data_x is not None:
        if data_x.shape != data.shape:
            raise ValueError("Incorrect shape for data_x")
        # interpolate output indices onto linear grid
        outinds = np.interp(out_x, data_x, np.arange(data.size))[newaxis,:]
    else:
        outinds = out_x[newaxis,:]

    # create the fourier kernel 
    kern=np.exp((-1j*2*pi)*freq*outinds)

    # the result is the dot product (sum along one axis) of the inverse fft of
    # the function and the kernel
    result = np.dot(ifftn(data),kern)

    if return_real:
        return result.real
    else:
        return result


