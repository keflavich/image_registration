import fast_ffts
import numpy as np

def shift(data, deltax, deltay, phase=0, nthreads=1, use_numpy_fft=False,
        return_abs=False, return_real=True):
    """
    FFT-based sub-pixel image shift
    Loosely based on:
    http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html
    But more directly and strictly, this is an implementation of the Fourier
    shift theorem.

    Will turn NaNs into zeros

    Parameters
    ----------
    """

    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)
    ny,nx = data.shape
    Nx = np.fft.ifftshift(np.linspace(-np.fix(nx/2),np.ceil(nx/2)-1,nx))
    Ny = np.fft.ifftshift(np.linspace(-np.fix(ny/2),np.ceil(ny/2)-1,ny))
    Nx,Ny = np.meshgrid(Nx,Ny)
    gg = ifftn( fftn(data)* np.exp(1j*2*np.pi*(-deltax*Nx/nx-deltay*Ny/ny)) * np.exp(-1j*phase) )
    if return_real:
        return np.real(gg)
    elif return_abs:
        return np.abs(gg)
    else:
        return gg

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
