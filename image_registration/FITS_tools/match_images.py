import numpy as np
from image_registration import chi2_shift,chi2_shift_iterzoom
try:
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
except ImportError:
    import pyfits
    import pywcs
from .load_header import load_data,load_header
from ..fft_tools.shift import shift2d

def project_to_header(fitsfile, header, use_montage=True, quiet=True,
                      **kwargs):
    """
    Light wrapper of montage with hcongrid as a backup

    Parameters
    ----------
        fitsfile : string
            a FITS file name
        header : pyfits.Header
            A pyfits Header instance with valid WCS to project to
        use_montage : bool
            Use montage or hcongrid (scipy's map_coordinates)
        quiet : bool
            Silence Montage's output

    Returns
    -------
        np.ndarray image projected to header's coordinates

    """
    try:
        import montage
        montageOK=True
    except ImportError:
        montageOK=False
    try:
        from hcongrid import hcongrid
        hcongridOK=True
    except ImportError:
        hcongridOK=False
    import tempfile

    if montageOK and use_montage:
        temp_headerfile = tempfile.NamedTemporaryFile()
        header.toTxtFile(temp_headerfile.name)

        if hasattr(fitsfile, 'writeto'):
            fitsobj = fitsfile
            fitsfileobj = tempfile.NamedTemporaryFile()
            fitsfile = fitsfileobj.name
            fitsobj.writeto(fitsfile)

        outfile = tempfile.NamedTemporaryFile()
        montage.wrappers.reproject(fitsfile,
                                   outfile.name,
                                   temp_headerfile.name,
                                   exact_size=True,
                                   silent_cleanup=quiet)
        image = pyfits.getdata(outfile.name)
        
        outfile.close()
        temp_headerfile.close()
        try:
            fitsfileobj.close()
        except NameError:
            pass
    elif hcongridOK:
        image = hcongrid(load_data(fitsfile),
                         load_header(fitsfile),
                         header)

    return image

def match_fits(fitsfile1, fitsfile2, header=None, sigma_cut=False,
               return_header=False, use_montage=False, **kwargs):
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
    use_montage: bool
        Use montage for the reprojection into the same pixel space?  Otherwise,
        use scipy.

    Returns
    -------
    image1,image2,[header] : 
        Two images projected into the same space, and optionally
        the header used to project them
    """

    if header is None:
        header = load_header(fitsfile1)
        image1 = load_data(fitsfile1)
    else: # project image 1 to input header coordinates
        image1 = project_to_header(fitsfile1, header, use_montage=use_montage)

    # project image 2 to image 1 coordinates
    image2_projected = project_to_header(fitsfile2, header, use_montage=use_montage)

    if image1.shape != image2_projected.shape:
        raise ValueError("Failed to reproject images to same shape.")

    if sigma_cut:
        corr_image1 = image1*(image1 > image1.std()*sigma_cut)
        corr_image2 = image2_projected*(image2_projected > image2_projected.std()*sigma_cut)
        OK = (corr_image1==corr_image1)*(corr_image2==corr_image2) 
        if (corr_image1[OK]*corr_image2[OK]).sum() == 0:
            print "Could not use sigma_cut of %f because it excluded all valid data" % sigma_cut
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
                  return_header=False, use_montage=False, **kwargs):
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
        Defaults to :func:`chi2_shift`
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
    use_montage: bool
        Use montage for the reprojection into the same pixel space?  Otherwise,
        use scipy.

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
    header : :class:`pyfits.Header` (only if `return_header` is True)
        The header the images have been projected to

    """
    proj_image1, proj_image2, header = match_fits(fitsfile1, fitsfile2,
            return_header=True, **kwargs)

    if errfile is not None:
        errimage = project_to_header(errfile, header, use_montage=use_montage, **kwargs)
    else:
        errimage = None

    xoff,yoff,exoff,eyoff = register_method(proj_image1, proj_image2,
            err=errimage, return_error=True, **kwargs)
    
    wcs = pywcs.WCS(header)
    try:
        cdelt = wcs.wcs.cd.diagonal()
    except AttributeError:
        cdelt = wcs.wcs.cdelt
    #print "CDELT: ",cdelt
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
