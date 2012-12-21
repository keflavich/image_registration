import fast_ffts
import numpy as np

def shift2d(data, deltax, deltay, phase=0, nthreads=1, use_numpy_fft=False,
        return_abs=False, return_real=True):
    """
    2D version: obsolete - use ND version instead
    (though it's probably easier to parse the source of this one)

    FFT-based sub-pixel image shift.
    Will turn NaNs into zeros

    Shift Theorem:

    .. math::
        FT[f(t-t_0)](x) = e^{-2 \pi i x t_0} F(x)


    Parameters
    ----------
    data : np.ndarray
        2D image
    phase : float
        Phase, in radians
    """

    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)
    ny,nx = data.shape

    xfreq = deltax * np.fft.fftfreq(nx)[np.newaxis,:]
    yfreq = deltay * np.fft.fftfreq(ny)[:,np.newaxis]
    freq_grid = xfreq + yfreq

    kernel = np.exp(-1j*2*np.pi*freq_grid-1j*phase)

    result = ifftn( fftn(data) * kernel )


    if return_real:
        return np.real(result)
    elif return_abs:
        return np.abs(result)
    else:
        return result

def shiftnd(data, offset, phase=0, nthreads=1, use_numpy_fft=False,
        return_abs=False, return_real=True):
    """
    FFT-based sub-pixel image shift.
    Will turn NaNs into zeros

    Shift Theorem:

    .. math::
        FT[f(t-t_0)](x) = e^{-2 \pi i x t_0} F(x)


    Parameters
    ----------
    data : np.ndarray
        Data to shift
    offset : (int,)*ndim
        Offsets in each direction.  Must be iterable.
    phase : float
        Phase, in radians

    Other Parameters
    ----------------
    use_numpy_fft : bool
        Force use numpy's fft over fftw?  (only matters if you have fftw
        installed)
    nthreads : bool
        Number of threads to use for fft (only matters if you have fftw
        installed)
    return_real : bool
        Return the real component of the shifted array
    return_abs : bool
        Return the absolute value of the shifted array

    Returns
    -------
    The input array shifted by offsets
    """

    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)

    freq_grid = np.sum(
        [off*np.fft.fftfreq(nx)[ 
            [np.newaxis]*(data.ndim-dim-1) + [slice(None)] + [np.newaxis]*dim]
            for dim,(off,nx) in enumerate(zip(offset,data.shape))],
        axis=0)

    kernel = np.exp(-1j*2*np.pi*freq_grid-1j*phase)

    result = ifftn( fftn(data) * kernel )

    if return_real:
        return np.real(result)
    elif return_abs:
        return np.abs(result)
    else:
        return result

if __name__ == "__main__":
    # A visual breakdown of the Fourier shift theorem
    # Lecture: http://www.cs.unm.edu/~williams/cs530/theorems6.pdf
    import pylab as pl

    xx,yy = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
    # a Gaussian image
    data = np.exp(-(xx**2+yy**2)/2.)

    pl.figure()
    pl.imshow(data)
    pl.title("Gaussian")

    ny,nx = data.shape
    Nx = np.fft.ifftshift(np.linspace(-np.fix(nx/2),np.ceil(nx/2)-1,nx))
    Ny = np.fft.ifftshift(np.linspace(-np.fix(ny/2),np.ceil(ny/2)-1,ny))

    pl.figure()
    pl.plot(Nx,label='Nx')
    pl.plot(Ny,label='Ny')
    pl.legend(loc='best')

    Nx,Ny = np.meshgrid(Nx,Ny)
    pl.figure()
    pl.subplot(121)
    pl.imshow(Nx)
    pl.title('Nx')
    pl.subplot(122)
    pl.imshow(Ny)
    pl.title('Ny')

    deltax,deltay = 1.5,-1.5 # 22.5,30.3

    kernel = np.exp(1j*2*np.pi*(-deltax*Nx/nx-deltay*Ny/ny))
    pl.figure()
    pl.subplot(131)
    pl.imshow(kernel.real)
    pl.title("kernel.real")
    pl.colorbar()
    pl.subplot(132)
    pl.imshow(kernel.imag)
    pl.title("kernel.imag")
    pl.colorbar()
    pl.subplot(133)
    pl.imshow(np.abs(kernel))
    pl.title("abs(kernel)")
    pl.colorbar()


    fftn,ifftn = fast_ffts.get_ffts(nthreads=4, use_numpy_fft=False)
    phase = 0
    gg = ifftn( fftn(data)* kernel * np.exp(-1j*phase) )

    pl.figure()
    pl.subplot(121)
    pl.imshow(gg.real)
    pl.title("gg.real")
    pl.subplot(122)
    pl.imshow(gg.imag)
    pl.title("gg.imag")
