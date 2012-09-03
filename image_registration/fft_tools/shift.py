import fast_ffts
import numpy as np

def shift(data, deltax, deltay, phase=0, nthreads=1, use_numpy_fft=False,
        return_abs=True):
    """
    FFT-based sub-pixel image shift
    http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

    Will turn NaNs into zeros
    """

    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)
    ny,nx = data.shape
    Nx = np.fft.ifftshift(np.linspace(-np.fix(nx/2),np.ceil(nx/2)-1,nx))
    Ny = np.fft.ifftshift(np.linspace(-np.fix(ny/2),np.ceil(ny/2)-1,ny))
    Nx,Ny = np.meshgrid(Nx,Ny)
    gg = ifftn( fftn(data)* np.exp(1j*2*np.pi*(-deltax*Nx/nx-deltay*Ny/ny)) * np.exp(-1j*phase) )
    if return_abs:
        return np.abs(gg)
    else:
        return gg
