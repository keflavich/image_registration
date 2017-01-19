import numpy as np
from ..chi2_shifts import chi2_shift_iterzoom
import astropy.wcs as pywcs
from .load_header import load_data,load_header
from ..fft_tools.shift import shift2d

def project_to_header(fitsfile, header, **kwargs):
    """
    Reproject an image to a header.  Simple wrapper of
    reproject.reproject_interp

    Parameters
    ----------
    fitsfile : string
        a FITS file name
    header : pyfits.Header
        A pyfits Header instance with valid WCS to project to
    quiet : bool
        Silence Montage's output

    Returns
    -------
    np.ndarray image projected to header's coordinates

    """
    import reproject
    return reproject.reproject_interp(fitsfile, header, **kwargs)[0]

def match_fits(fitsfile1, fitsfile2, header=None, sigma_cut=False,
               return_header=False, **kwargs):
    """
    Project one FITS file into another's coordinates
    If sigma_cut is used, will try to find only regions that are significant
    in both images using the standard deviation

    Parameters
    ----------
    fitsfile1: str
        Reference fits file name
    fitsfile2: str
        Offset fits file name
    header: pyfits.Header
        Optional - can pass a header to projet both images to
    sigma_cut: bool or int
        Perform a sigma-cut on the returned images at this level

    Returns
    -------
    image1,image2,[header] : Two images projected into the same space, and
    optionally the header used to project them
    """

    if header is None:
        header = load_header(fitsfile1)
        image1 = load_data(fitsfile1)
    else: # project image 1 to input header coordinates
        image1 = project_to_header(fitsfile1, header)

    # project image 2 to image 1 coordinates
    image2_projected = project_to_header(fitsfile2, header)

    if image1.shape != image2_projected.shape:
        raise ValueError("Failed to reproject images to same shape.")

    if sigma_cut:
        corr_image1 = image1*(image1 > image1.std()*sigma_cut)
        corr_image2 = image2_projected*(image2_projected > image2_projected.std()*sigma_cut)
        OK = (corr_image1==corr_image1)*(corr_image2==corr_image2) 
        if (corr_image1[OK]*corr_image2[OK]).sum() == 0:
            print("Could not use sigma_cut of %f because it excluded all valid data" % sigma_cut)
            corr_image1 = image1
            corr_image2 = image2_projected
    else:
        corr_image1 = image1
        corr_image2 = image2_projected

    returns = corr_image1, corr_image2
    if return_header:
        returns = returns + (header,)
    return returns

def register_fits(fitsfile1, fitsfile2, errfile=None, return_error=True,
                  register_method=chi2_shift_iterzoom,
                  return_cropped_images=False, return_shifted_image=False,
                  return_header=False, **kwargs):
    """
    Determine the shift between two FITS images using the cross-correlation
    technique.  Requires montage or hcongrid.

    kwargs are passed to :func:`register_method`

    Parameters
    ----------
    fitsfile1: str
        Reference fits file name
    fitsfile2: str
        Offset fits file name
    errfile : str [optional]
        An error image, intended to correspond to fitsfile2
    register_method : function
        Can be any of the shift functions in :mod:`image_registration`.
        Defaults to :func:`chi2_shift_iterzoom`
    return_errors: bool
        Return the errors on each parameter in addition to the fitted offset
    return_cropped_images: bool
        Returns the images used for the analysis in addition to the measured
        offsets
    return_shifted_images: bool
        Return image 2 shifted into image 1's space
    return_header : bool
        Return the header the images have been projected to
    quiet: bool
        Silence messages?
    sigma_cut: bool or int
        Perform a sigma-cut before cross-correlating the images to minimize
        noise correlation?

    Returns
    -------
    xoff,yoff : (float,float)
        pixel offsets
    xoff_wcs,yoff_wcs : (float,float)
        world coordinate offsets
    exoff,eyoff : (float,float) (only if `return_errors` is True)
        Standard error on the fitted pixel offsets
    exoff_wcs,eyoff_wcs : (float,float) (only if `return_errors` is True)
        Standard error on the fitted world coordinate offsets
    proj_image1, proj_image2 : (ndarray,ndarray) (only if `return_cropped_images` is True)
        The images projected into the same coordinates
    shifted_image2 : ndarray (if `return_shifted_image` is True)
        The second image projected *and shifted* to match image 1.
    header : :class:`pyfits.Header` (only if `return_header` is True)
        The header the images have been projected to

    """
    proj_image1, proj_image2, header = match_fits(fitsfile1, fitsfile2,
                                                  return_header=True, **kwargs)

    if errfile is not None:
        errimage = project_to_header(errfile, header, **kwargs)
    else:
        errimage = None

    xoff,yoff,exoff,eyoff = register_method(proj_image1, proj_image2,
                                            err=errimage, return_error=True,
                                            **kwargs)
    
    wcs = pywcs.WCS(header)
    try:
        cdelt = wcs.wcs.cd.diagonal()
    except AttributeError:
        cdelt = wcs.wcs.cdelt
    xoff_wcs,yoff_wcs = np.array([xoff,yoff])*cdelt
    exoff_wcs,eyoff_wcs = np.array([exoff,eyoff])*cdelt
    #try:
    #    xoff_wcs,yoff_wcs = np.inner( np.array([[xoff,0],[0,yoff]]), wcs.wcs.cd )[[0,1],[0,1]]
    #except AttributeError:
    #    xoff_wcs,yoff_wcs = 0,0

    returns = xoff,yoff,xoff_wcs,yoff_wcs
    if return_error:
        returns = returns + (exoff,eyoff,exoff_wcs,eyoff_wcs)
    if return_cropped_images:
        returns = returns + (proj_image1,proj_image2)
    if return_shifted_image:
        shifted_im2 = shift2d(proj_image2, -xoff, -yoff)
        returns = returns + (shifted_im2,)
    if return_header:
        returns = returns + (header,)
    return returns
