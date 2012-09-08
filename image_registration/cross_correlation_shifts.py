try: 
    from AG_fft_tools import correlate2d,fast_ffts
except ImportError:
    from image_registration.fft_tools import correlate2d,fast_ffts
import warnings
import numpy as np

__all__ = ['cross_correlation_shifts','cross_correlation_shifts_FITS']

def cross_correlation_shifts(image1, image2, errim1=None, errim2=None,
        maxoff=None, verbose=False, gaussfit=False, return_error=False,
        zeromean=True, **kwargs):
    """ Use cross-correlation and a 2nd order taylor expansion to measure the
    offset between two images

    Given two images, calculate the amount image2 is offset from image1 to
    sub-pixel accuracy using 2nd order taylor expansion.

    Parameters
    ----------
    image1: np.ndarray
        The reference image
    image2: np.ndarray
        The offset image.  Must have the same shape as image1
    errim1: np.ndarray [optional]
        The pixel-by-pixel error on the reference image
    errim2: np.ndarray [optional]
        The pixel-by-pixel error on the offset image.  
    maxoff: int
        Maximum allowed offset (in pixels).  Useful for low s/n images that you
        know are reasonably well-aligned, but might find incorrect offsets due to 
        edge noise
    zeromean : bool
        Subtract the mean from each image before performing cross-correlation?
    verbose: bool
        Print out extra messages?
    gaussfit : bool
        Use a Gaussian fitter to fit the peak of the cross-correlation?
    return_error: bool
        Return an estimate of the error on the shifts.  WARNING: I still don't
        understand how to make these agree with simulations.
        The analytic estimate comes from
        http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
        At high signal-to-noise, the analytic version overestimates the error
        by a factor of about 1.8, while the gaussian version overestimates
        error by about 1.15.  At low s/n, they both UNDERestimate the error.
        The transition zone occurs at a *total* S/N ~ 1000 (i.e., the total
        signal in the map divided by the standard deviation of the map - 
        it depends on how many pixels have signal)

    **kwargs are passed to correlate2d, which in turn passes them to convolve.
    The available options include image padding for speed and ignoring NaNs.

    References
    ----------
    From http://solarmuri.ssl.berkeley.edu/~welsch/public/software/cross_cor_taylor.pro

    Examples
    --------
    >>> import numpy as np
    >>> im1 = np.zeros([10,10])
    >>> im2 = np.zeros([10,10])
    >>> im1[4,3] = 1
    >>> im2[5,5] = 1
    >>> import AG_image_tools
    >>> yoff,xoff = AG_image_tools.cross_correlation_shifts(im1,im2)
    >>> im1_aligned_to_im2 = np.roll(np.roll(im1,int(yoff),1),int(xoff),0)
    >>> assert (im1_aligned_to_im2-im2).sum() == 0
    

    """

    if not image1.shape == image2.shape:
        raise ValueError("Images must have same shape.")

    if zeromean:
        image1 = image1 - (image1[image1==image1].mean())
        image2 = image2 - (image2[image2==image2].mean())

    image1 = np.nan_to_num(image1)
    image2 = np.nan_to_num(image2)

    quiet = kwargs.pop('quiet') if 'quiet' in kwargs else not verbose
    ccorr = (correlate2d(image1,image2,quiet=quiet,**kwargs) / image1.size)
    # allow for NaNs set by convolve (i.e., ignored pixels)
    ccorr[ccorr!=ccorr] = 0
    if ccorr.shape != image1.shape:
        raise ValueError("Cross-correlation image must have same shape as input images.  This can only be violated if you pass a strange kwarg to correlate2d.")

    ylen,xlen = image1.shape
    xcen = xlen/2-(1-xlen%2) 
    ycen = ylen/2-(1-ylen%2) 

    if ccorr.max() == 0:
        warnings.warn("WARNING: No signal found!  Offset is defaulting to 0,0")
        return 0,0

    if maxoff is not None:
        if verbose: print "Limiting maximum offset to %i" % maxoff
        subccorr = ccorr[ycen-maxoff:ycen+maxoff+1,xcen-maxoff:xcen+maxoff+1]
        ymax,xmax = np.unravel_index(subccorr.argmax(), subccorr.shape)
        xmax = xmax+xcen-maxoff
        ymax = ymax+ycen-maxoff
    else:
        ymax,xmax = np.unravel_index(ccorr.argmax(), ccorr.shape)
        subccorr = ccorr

    if return_error:
        if errim1 is None:
            errim1 = np.ones(ccorr.shape) * image1[image1==image1].std() 
        if errim2 is None:
            errim2 = np.ones(ccorr.shape) * image2[image2==image2].std() 
        eccorr =( (correlate2d(errim1**2, image2**2,quiet=quiet,**kwargs)+
                   correlate2d(errim2**2, image1**2,quiet=quiet,**kwargs))**0.5 
                   / image1.size)
        if maxoff is not None:
            subeccorr = eccorr[ycen-maxoff:ycen+maxoff+1,xcen-maxoff:xcen+maxoff+1]
        else:
            subeccorr = eccorr

    if gaussfit:
        try: 
            from agpy import gaussfitter
        except ImportError:
            raise ImportError("Couldn't import agpy.gaussfitter; try using cross_correlation_shifts with gaussfit=False")
        if return_error:
            pars,epars = gaussfitter.gaussfit(subccorr,err=subeccorr,return_all=True)
            exshift = epars[2]
            eyshift = epars[3]
        else:
            pars,epars = gaussfitter.gaussfit(subccorr,return_all=True)
        xshift = maxoff - pars[2] if maxoff is not None else xcen - pars[2]
        yshift = maxoff - pars[3] if maxoff is not None else ycen - pars[3]
        if verbose: 
            print "Gaussian fit pars: ",xshift,yshift,epars[2],epars[3],pars[4],pars[5],epars[4],epars[5]

    else:

        xshift_int = xmax-xcen
        yshift_int = ymax-ycen

        local_values = ccorr[ymax-1:ymax+2,xmax-1:xmax+2]

        d1y,d1x = np.gradient(local_values)
        d2y,d2x,dxy = second_derivative(local_values)

        fx,fy,fxx,fyy,fxy = d1x[1,1],d1y[1,1],d2x[1,1],d2y[1,1],dxy[1,1]

        shiftsubx=(fyy*fx-fy*fxy)/(fxy**2-fxx*fyy)
        shiftsuby=(fxx*fy-fx*fxy)/(fxy**2-fxx*fyy)

        xshift = -(xshift_int+shiftsubx)
        yshift = -(yshift_int+shiftsuby)

        # http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
        # Zucker error

        if return_error:
            #acorr1 = (correlate2d(image1,image1,quiet=quiet,**kwargs) / image1.size)
            #acorr2 = (correlate2d(image2,image2,quiet=quiet,**kwargs) / image2.size)
            #ccorrn = ccorr / eccorr**2 / ccorr.size #/ (errim1.mean()*errim2.mean()) #/ eccorr**2
            normalization = 1. / ((image1**2).sum()/image1.size) / ((image2**2).sum()/image2.size) 
            ccorrn = ccorr * normalization
            exshift = (np.abs(-1 * ccorrn.size * fxx*normalization/ccorrn[ymax,xmax] *
                    (ccorrn[ymax,xmax]**2/(1-ccorrn[ymax,xmax]**2)))**-0.5) 
            eyshift = (np.abs(-1 * ccorrn.size * fyy*normalization/ccorrn[ymax,xmax] *
                    (ccorrn[ymax,xmax]**2/(1-ccorrn[ymax,xmax]**2)))**-0.5) 
            if np.isnan(exshift):
                raise ValueError("Error: NAN error!")

    if return_error:
        return xshift,yshift,exshift,eyshift
    else:
        return xshift,yshift

def second_derivative(image):
    """
    Compute the second derivative of an image
    The derivatives are set to zero at the edges

    Parameters
    ----------
    image: np.ndarray

    Returns
    -------
    d/dx^2, d/dy^2, d/dxdy
    All three are np.ndarrays with the same shape as image.

    """
    shift_right = np.roll(image,1,1)
    shift_right[:,0] = 0
    shift_left = np.roll(image,-1,1)
    shift_left[:,-1] = 0
    shift_down = np.roll(image,1,0)
    shift_down[0,:] = 0
    shift_up = np.roll(image,-1,0)
    shift_up[-1,:] = 0

    shift_up_right = np.roll(shift_up,1,1)
    shift_up_right[:,0] = 0
    shift_down_left = np.roll(shift_down,-1,1)
    shift_down_left[:,-1] = 0
    shift_down_right = np.roll(shift_right,1,0)
    shift_down_right[0,:] = 0
    shift_up_left = np.roll(shift_left,-1,0)
    shift_up_left[-1,:] = 0

    dxx = shift_right+shift_left-2*image
    dyy = shift_up   +shift_down-2*image
    dxy=0.25*(shift_up_right+shift_down_left-shift_up_left-shift_down_right)

    return dxx,dyy,dxy

def cross_correlation_shifts_FITS(fitsfile1, fitsfile2,
        return_cropped_images=False, quiet=True, sigma_cut=False,
        register_method=cross_correlation_shifts, **kwargs):
    """
    Determine the shift between two FITS images using the cross-correlation
    technique.  Requires montage or hcongrid.

    Parameters
    ----------
    fitsfile1: str
        Reference fits file name
    fitsfile2: str
        Offset fits file name
    return_cropped_images: bool
        Returns the images used for the analysis in addition to the measured
        offsets
    quiet: bool
        Silence messages?
    sigma_cut: bool or int
        Perform a sigma-cut before cross-correlating the images to minimize
        noise correlation?
    """
    import montage
    try:
        import astropy.io.fits as pyfits
        import astropy.wcs as pywcs
    except ImportError:
        import pyfits
        import pywcs
    import tempfile

    header = pyfits.getheader(fitsfile1)
    temp_headerfile = tempfile.NamedTemporaryFile()
    header.toTxtFile(temp_headerfile.name)

    outfile = tempfile.NamedTemporaryFile()
    montage.wrappers.reproject(fitsfile2, outfile.name, temp_headerfile.name, exact_size=True, silent_cleanup=quiet)
    image2_projected = pyfits.getdata(outfile.name)
    image1 = pyfits.getdata(fitsfile1)
    
    outfile.close()
    temp_headerfile.close()

    if image1.shape != image2_projected.shape:
        raise ValueError("montage failed to reproject images to same shape.")

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

    verbose = kwargs.pop('verbose') if 'verbose' in kwargs else not quiet
    xoff,yoff = register_method(corr_image1, corr_image2, verbose=verbose,**kwargs)
    
    wcs = pywcs.WCS(header)
    try:
        xoff_wcs,yoff_wcs = np.inner( np.array([[xoff,0],[0,yoff]]), wcs.wcs.cd )[[0,1],[0,1]]
    except AttributeError:
        xoff_wcs,yoff_wcs = 0,0

    if return_cropped_images:
        return xoff,yoff,xoff_wcs,yoff_wcs,image1,image2_projected
    else:
        return xoff,yoff,xoff_wcs,yoff_wcs
    


