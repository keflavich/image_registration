from . import fast_ffts
import numpy as np

def fourier_interp1d(data, out_x, data_x=None, nthreads=1, use_numpy_fft=False,
                     return_real=True):
    """
    Use the fourier scaling theorem to interpolate (or extrapolate, without
    raising any exceptions) data.

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

def fourier_interp2d(data, outinds, nthreads=1, use_numpy_fft=False,
                     return_real=True):
    """
    Use the fourier scaling theorem to interpolate (or extrapolate, without raising
    any exceptions) data.

    Parameters
    ----------
    data : ndarray
        The data values of the array to interpolate
    outinds : ndarray
        The coordinate axis values along which the data should be interpolated
        CAN BE: `ndim x [n,m,...]` like np.indices OR (less memory intensive,
        more processor intensive) `([n],[m],...)`
    """

    # load fft
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if hasattr(outinds,'ndim') and outinds.ndim not in (data.ndim+1,data.ndim):
        raise ValueError("Must specify an array of output indices with # of dimensions = input # of dims + 1")
    elif len(outinds) != data.ndim:
        raise ValueError("outind array must have an axis for each dimension")

    imfft = ifftn(data)

    freqY = np.fft.fftfreq(data.shape[0])
    if hasattr(outinds,'ndim') and outinds.ndim == 3:
        # if outinds = np.indices(shape), we extract just lines along each index
        indsY = freqY[np.newaxis,:]*outinds[0,:,0][:,np.newaxis]
    else:
        indsY = freqY[np.newaxis,:]*np.array(outinds[0])[:,np.newaxis]
    kerny=np.exp((-1j*2*np.pi)*indsY)

    freqX = np.fft.fftfreq(data.shape[1])
    if hasattr(outinds,'ndim') and outinds.ndim == 3:
        # if outinds = np.indices(shape), we extract just lines along each index
        indsX = freqX[:,np.newaxis]*outinds[1,0,:][np.newaxis,:]
    else:
        indsX = freqX[:,np.newaxis]*np.array(outinds[1])[np.newaxis,:]
    kernx=np.exp((-1j*2*np.pi)*indsX)

    result = np.dot(np.dot(kerny, imfft), kernx)

    if return_real:
        return result.real
    else:
        return result



def fourier_interpnd(data, outinds, nthreads=1, use_numpy_fft=False,
                     return_real=True):
    """
    Use the fourier scaling theorem to interpolate (or extrapolate, without raising
    any exceptions) data.
    * DOES NOT WORK FOR ANY BUT 2 DIMENSIONS *

    Parameters
    ----------
    data : ndarray
        The data values of the array to interpolate
    outinds : ndarray
        The coordinate axis values along which the data should be interpolated
        CAN BE `ndim x [n,m,...]` like np.indices OR (less memory intensive,
        more processor intensive) `([n],[m],...)`
    """

    # load fft
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if hasattr(outinds,'ndim') and outinds.ndim not in (data.ndim+1,data.ndim):
        raise ValueError("Must specify an array of output indices with # of dimensions = input # of dims + 1")
    elif len(outinds) != data.ndim:
        raise ValueError("outind array must have an axis for each dimension")

    imfft = ifftn(data)
    result = imfft

    for dim,dimsize in enumerate(data.shape):

        # specify fourier frequencies
        freq = np.fft.fftfreq(dimsize)

        # have to cleverly specify frequency dimensions for the dot
        # frequency is the axis that will get summed over
        freqdims = [None]*(dim) + [slice(None)] + [None]*(data.ndim-1-dim)
        inddims = [None]*(data.ndim-1-dim) + [slice(None)] + [None]*(dim)

        # create the fourier kernel 
        if hasattr(outinds,'ndim') and outinds.ndim == data.ndim+1:
            # if outinds = np.indices(shape), we extract just lines along each index
            outslice = [dim] + [0]*dim + [slice(None)] + [0]*(data.ndim-1-dim)
            inds = freq[freqdims]*outinds[outslice][inddims]
        # un-pythonic? elif hasattr(outinds[dim],'ndim') and outinds[dim].ndim == 1:
        else:
            inds = freq[freqdims]*np.array(outinds[dim])[inddims]

        kern=np.exp((-1j*2*np.pi)*inds)

        # the result is the dot product (sum along one axis) of the inverse fft of
        # the function and the kernel
        # first time: dim = 0   ndim-1-dim = 1
        # second time: dim = 1  ndim-1-dim = 0
        result = np.dot(result.swapaxes(dim,-1),kern.swapaxes(data.ndim-1-dim,-1)).swapaxes(dim,-1)

    if return_real:
        return result.real
    else:
        return result
