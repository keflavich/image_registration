try: 
    from AG_fft_tools import correlate1d,fast_ffts,dftups1d,upsample_image
except ImportError:
    from image_registration.fft_tools import correlate1d,fast_ffts,dftups1d,upsample_image
import warnings
import numpy as np

__all__ = ['chi2_shift_1d']

def chi2_shift_1d(sp1, sp2, err=None, upsample_factor=10, boundary='wrap',
        nthreads=1, use_numpy_fft=False, zeromean=False, nfitted=1, verbose=False,
        return_error=True, return_chi2array=False, max_auto_size=512,
        max_nsig=1.1):
    """
    Find the offsets between image 1 and image 2 using the DFT upsampling method
    (http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html)
    combined with :math:`chi^2` to measure the errors on the fit

    .. math::
            \chi^2 & = & \Sigma_{ij} \\frac{(X_{ij}-Y_{ij})^2}{\sigma_{ij}^2} \\\\
                   & = & \Sigma_{ij} \left[ X_{ij}^2/\sigma_{ij}^2 - 2X_{ij}Y_{ij}/\sigma_{ij} + Y_{ij}^2 \\right]  \\\\

    .. math::
            \Sigma_{ij}[x,y] \\frac{X_{ij}^2}{\sigma_{ij}^2} & = & X/dx * X/dx \\\\
            \Sigma_{ij}[x,y] \\frac{X_{ij}Y_{ij}}{\sigma_{ij}} & = & X/dx * Y \\\\
            \Sigma_{ij}[x,y] Y_{ij}^2 & = &  Y * Y

    
    
    Parameters
    ----------
    sp1 : np.ndarray
    sp2 : np.ndarray
        The 1D-images (spectra?) to register. 
    err : np.ndarray
        Per-pixel error in image 2
    boundary : 'wrap','constant','reflect','nearest'
        Option to pass to map_coordinates for determining what to do with
        shifts outside of the boundaries.  
    upsample_factor : int or 'auto'
        upsampling factor; governs accuracy of fit (1/usfac is best accuracy)
        (can be "automatically" determined based on chi^2 error)
    return_error : bool
        Returns the "fit error" (1-sigma in x and y) based on the delta-chi2
        values
    return_chi2_array : bool
        Returns the x and y shifts and the chi2 as a function of those shifts
        in addition to other returned parameters.  i.e., the last return from
        this function will be a tuple (x, y, chi2)
    zeromean : bool
        Subtract the mean from the images before cross-correlating?  If no, you
        may get a 0,0 offset because the DC levels are strongly correlated.
    verbose : bool
        Print error message if upsampling factor is inadequate to measure errors
    use_numpy_fft : bool
        Force use numpy's fft over fftw?  (only matters if you have fftw
        installed)
    nthreads : bool
        Number of threads to use for fft (only matters if you have fftw
        installed)
    nfitted : int
        number of degrees of freedom in the fit (used for chi^2 computations).
        Should probably always be 1.
    max_auto_size : int
        Maximum zoom image size to create when using auto-upsampling


    Returns
    -------
    dx,dy : float,float
        Measures the amount sp2 is offset from sp1 (i.e., shift sp2 by -1 *
        these #'s to match sp1)

        .. todo:: CHECK THIS 

    Examples
    --------
    >>> # Create a 1d array
    >>> image = np.random.randn(500)
    >>> # shift it 
    >>> shifted = np.roll(image,12,0)
    >>> # determine shift
    >>> import image_registration
    >>> dx,dy,edx,edy = image_registration.chi2_shift(image, shifted, upsample_factor='auto')
    >>> # Check that the shift is correct
    >>> print "dx - fitted dx = ",dx-5," error: ",edx
    >>> print "dy - fitted dy = ",dy-12," error: ",edy
    >>> # that example was boring; instead let's do one with a non-int shift
    >>> shifted2 = image_registration.fft_tools.shift(image,3.665,-4.25)
    >>> dx2,dy2,edx2,edy2 = image_registration.chi2_shift(image, shifted2, upsample_factor='auto')
    >>> print "dx - fitted dx = ",dx2-3.665," error: ",edx2
    >>> print "dy - fitted dy = ",dy2-(-4.25)," error: ",edy2
    
    .. todo:: understand numerical error in fft-shifted version

    """
    if not sp1.shape == sp2.shape:
        raise ValueError("Images must have same shape.")

    if zeromean:
        sp1 = sp1 - (sp1[sp1==sp1].mean())
        sp2 = sp2 - (sp2[sp2==sp2].mean())

    sp1 = np.nan_to_num(sp1)
    sp2 = np.nan_to_num(sp2)

    xc = correlate1d(sp1,sp2, boundary=boundary)
    if err is not None:
        err = np.nan_to_num(err)
        err_ac = correlate1d(err,err, boundary=boundary)
        err2sum = (err**2).sum()
    else:
        err_ac = xc.size - nfitted
        err2sum = xc.size - nfitted
    ac1peak = (sp1**2).sum()
    ac2peak = (sp2**2).sum()
    chi2n = (ac1peak/err2sum - 2*xc/err_ac + ac2peak/err2sum) 
    xmax = chi2n.argmin()

    xlen = sp1.size
    xcen = xlen/2-(1-xlen%2) 

    # original shift calculation
    # shift sp2 by these numbers to get sp1
    xshift = xmax-xcen

    # below is sub-pixel zoom-in stuff

    # find delta-chi^2 limiting values for varying DOFs
    try:
        import scipy.stats
        # 1,2,3-sigma delta-chi2 levels
        m1 = scipy.stats.chi2.ppf( 1-scipy.stats.norm.sf(1)*2, nfitted )
        m2 = scipy.stats.chi2.ppf( 1-scipy.stats.norm.sf(2)*2, nfitted )
        m3 = scipy.stats.chi2.ppf( 1-scipy.stats.norm.sf(3)*2, nfitted )
        m_auto = scipy.stats.chi2.ppf( 1-scipy.stats.norm.sf(max_nsig)*2, nfitted )
    except ImportError:
        # assume m=2 (2 degrees of freedom)
        m1 = 2.2957489288986364
        m2 = 6.1800743062441734 
        m3 = 11.829158081900793
        m_auto = 2.6088233328527037 # slightly >1 sigma

    # biggest scale = where chi^2/n ~ 9 or 11.8 for M=2?
    if upsample_factor=='auto':
        # deltachi2 is not reduced deltachi2
        deltachi2_lowres = (chi2n - chi2n.min())*(xc.size-nfitted-1)
        if verbose:
            print "Minimum chi2n: %g   Max delta-chi2 (lowres): %g  Min delta-chi2: %g" % (chi2n.min(),deltachi2_lowres.max(),deltachi2_lowres[deltachi2_lowres>0].min())
        sigmamax_area = deltachi2_lowres<m_auto
        if sigmamax_area.sum() > 1:
            xx = np.arange(sigmamax_area.size)
            xvals = xx[sigmamax_area]
            xvrange = xvals.max()-xvals.min()
            size = xvrange
        else:
            size = 1
        upsample_factor = max_auto_size/2. / size
        if upsample_factor < 1:
            upsample_factor = 1
        s1 = s2 = max_auto_size
        # zoom factor = s1 / upsample_factor = 2*size
        zoom_factor = 2.*size
        if verbose:
            print "Selected upsample factor %0.1f for image size %i and zoom factor %0.1f (max-sigma range was %i for area %i)" % (upsample_factor, s1, zoom_factor, size, sigmamax_area.sum())
    else:
        s1 = s2 = sp1.size

        zoom_factor = s1/upsample_factor
        if zoom_factor <= 1:
            zoom_factor = 2
            s1 = zoom_factor*upsample_factor
            s2 = zoom_factor*upsample_factor

    # import fft's
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    # pilfered from dftregistration (hence the % comments)
    dftshift = np.trunc(np.ceil(upsample_factor*zoom_factor)/2); #% Center of output array at dftshift+1
    xc_ups = dftups1d(fftn(sp2)*np.conj(fftn(sp1)), s1, usfac=upsample_factor,
            roff=dftshift-xshift*upsample_factor) / (sp1.size) #*upsample_factor**2)
    if err is not None:
        err_ups = upsample_image(err_ac, output_size=s1,
                upsample_factor=upsample_factor, xshift=xshift)
    else:
        err_ups = 1
    chi2n_ups = (ac1peak/err2sum-2*np.abs(xc_ups)/np.abs(err_ups)+ac2peak/err2sum)#*(xc.size-nfitted)
    # deltachi2 is not reduced deltachi2
    deltachi2 = (chi2n_ups - chi2n_ups.min())*(xc.size-nfitted-1)
    if verbose:
        print "Minimum chi2n_ups: %g   Max delta-chi2: %g  Min delta-chi2: %g" % (chi2n_ups.min(),deltachi2.max(),deltachi2[deltachi2>0].min())

    xx = np.arange(s1)
    xshifts_corrections = (xx-dftshift)/upsample_factor

    sigma1_area = deltachi2<m1
    # optional...?
    #sigma2_area = deltachi2<m2
    #x_sigma2 = xshifts_corrections[sigma2_area]
    #y_sigma2 = yshifts_corrections[sigma2_area]
    #sigma3_area = deltachi2<m3
    #x_sigma3 = xshifts_corrections[sigma3_area]
    #y_sigma3 = yshifts_corrections[sigma3_area]

    # upsampled maximum - correction
    upsxmax = chi2n_ups.argmin()

    if sigma1_area.sum() <= 1:
        if verbose:
            print "Cannot estimate errors: need higher upsample factor.  Sigmamax_area=%i" % (sigmamax_area.sum())
        errx_low = errx_high = 1./upsample_factor
    else: # compute 1-sigma errors
        x_sigma1 = xshifts_corrections[sigma1_area]

        errx_low = (upsxmax-dftshift)/upsample_factor - x_sigma1.min()
        errx_high = x_sigma1.max() - (upsxmax-dftshift)/upsample_factor
        if verbose:
            print "Found %i pixels within the 1-sigma region. xmin,ymin: %f,%f xmax,ymax: %g,%g" % (sigma1_area.sum(),x_sigma1.min(),y_sigma1.min(),x_sigma1.max(),y_sigma1.max())
        #from pylab import *
        #clf(); imshow(deltachi2); colorbar(); contour(sigma1_area,levels=[0.5],colors=['w'])
        #import pdb; pdb.set_trace()

    xshift_corr = xshift+(upsxmax-dftshift)/float(upsample_factor)
    if verbose > 1:
        #print ymax,xmax
        #print upsymax, upsxmax
        #print upsymax-dftshift, upsxmax-dftshift
        print "Correction: ",(upsxmax-dftshift)/float(upsample_factor)
        print "Chi2 1sig bounds:", x_sigma1.min(), x_sigma1.max()
        print errx_low,errx_high
        print "%0.3f +%0.3f -%0.3f   %0.3f +%0.3f -%0.3f" % (xshift_corr, errx_high, errx_low)
        #print ymax-ycen+upsymax/float(upsample_factor), xmax-xcen+upsxmax/float(upsample_factor)
        #print (upsymax-s1/2)/upsample_factor, (upsxmax-s2/2)/upsample_factor

    shift_xvals = xshifts_corrections+xshift

    returns = [-xshift_corr]
    if return_error:
        returns.append( (errx_low+errx_high)/2. )
    if return_chi2array:
        returns.append((shift_xvals,chi2n_ups))

    return returns


