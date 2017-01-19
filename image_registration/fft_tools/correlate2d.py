import numpy as np
from .convolve_nd import convolvend as convolve

__all__ = ['correlate2d']


def correlate2d(im1,im2, boundary='wrap', **kwargs):
    """
    Cross-correlation of two images of arbitrary size.  Returns an image
    cropped to the largest of each dimension of the input images

    Parameters
    ----------
    return_fft - if true, return fft(im1)*fft(im2[::-1,::-1]), which is the power
        spectral density
    fftshift - if true, return the shifted psd so that the DC component is in
        the center of the image
    pad - Default on.  Zero-pad image to the nearest 2^n
    crop - Default on.  Return an image of the size of the largest input image.
        If the images are asymmetric in opposite directions, will return the largest
        image in both directions.
    boundary: str, optional
        A flag indicating how to handle boundaries:
            * 'fill' : set values outside the array boundary to fill_value
                       (default)
            * 'wrap' : periodic boundary

    WARNING: Normalization may be arbitrary if you use the PSD
    """

    return convolve(np.conjugate(im1), im2[::-1, ::-1], normalize_kernel=False,
            boundary=boundary, ignore_edge_zeros=False, **kwargs)

