import fast_ffts
import numpy as np

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

    Other Parameters
    ----------------
    nthreads : int
        Number of threads for parallelized FFTs (if available)
    use_numpy_fft : bool
        Use the numpy version of the FFT before any others?  (Default is to use
        fftw3)

    Returns
    -------
    The real component of the interpolated 1D array, or the full complex array
    if return_real is False

    Raises
    ------
    ValueError if output indices are the wrong shape or the data X array is the
    wrong shape
    """

    # load fft
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    # specify fourier frequencies
    freq = np.fft.fftfreq(data.size)[:,np.newaxis]

    # reshape outinds
    if out_x.ndim != 1:
        raise ValueError("Must specify a 1-d array of output indices")

    if data_x is not None:
        if data_x.shape != data.shape:
            raise ValueError("Incorrect shape for data_x")
        # interpolate output indices onto linear grid
        outinds = np.interp(out_x, data_x, np.arange(data.size))[np.newaxis,:]
    else:
        outinds = out_x[np.newaxis,:]

    # create the fourier kernel 
    kern=np.exp((-1j*2*np.pi)*freq*outinds)

    # the result is the dot product (sum along one axis) of the inverse fft of
    # the function and the kernel
    result = np.dot(ifftn(data),kern)

    if return_real:
        return result.real
    else:
        return result


def fourier_interpnd(data, outinds, nthreads=1, use_numpy_fft=False,
        return_real=True):
    """
    Use the fourier scaling theorem to interpolate (or extrapolate, without raising
    any exceptions) data.

    Parameters
    ----------
    data : ndarray
        The Y-values of the array to interpolate
    outinds : ndarray
        The X-values along which the data should be interpolated
    """

    # load fft
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if outinds.ndim != data.ndim:
        raise ValueError("Must specify an array of output indices with same # of dimensions as input")

    imfft = ifftn(data)
    result = imfft

    for dim,dimsize in enumerate(data.shape):

        # specify fourier frequencies
        freq = np.fft.fftfreq(dimsize)

        # have to cleverly specify frequency dimensions for the dot
        # frequency is the axis that will get summed over
        freqdims = [None]*(data.ndim-2) + slice(None) + [None]

        # create the fourier kernel 
        kern=np.exp((-1j*2*np.pi)*freq[freqdims]*outinds.swapaxes(dim,-1))

        # the result is the dot product (sum along one axis) of the inverse fft of
        # the function and the kernel
        result = np.dot(result.swapaxes(dim,-1),kern)

    if return_real:
        return result.real
    else:
        return result


