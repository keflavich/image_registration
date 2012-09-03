from AG_fft_tools import correlate2d,fast_ffts
import warnings
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

def chi2(im1, im2, dx, dy, upsample=1):
    im1 = np.nan_to_num(im1)
    im2 = np.nan_to_num(im2)

    if upsample > 1:
        im1  = upsample_image(im1, upsample_factor=upsample, output_size=im1.shape, )
        im2s = upsample_image(im2, upsample_factor=upsample, output_size=im2.shape, xshift=-dx*upsample, yshift=-dy*upsample)
        #im2s = np.abs(shift(im2, -dx*upsample, -dy*upsample))

    else:
        im2s = np.abs(shift(im2, -dx, -dy))

    return ((im1-im2s)**2).sum()


def register(im1, im2, usfac=1, return_registered=False, return_error=False, zeromean=True, DEBUG=False, maxoff=None, nthreads=1, use_numpy_fft=False):
    """
    Sub-pixel image registration (see dftregistration for lots of details)

    Parameters
    ----------
    im1 : np.ndarray
    im2 : np.ndarray
        The images to register. 
    usfac : int
        upsampling factor; governs accuracy of fit (1/usfac is best accuracy)
    return_registered : bool
        Return the registered image as the last parameter
    return_error : bool
        Does nothing at the moment, but in principle should return the "fit
        error" (it does nothing because I don't know how to compute the "fit
        error")
    zeromean : bool
        Subtract the mean from the images before cross-correlating?  If no, you
        may get a 0,0 offset because the DC levels are strongly correlated.
    maxoff : int
        Maximum allowed offset to measure (setting this helps avoid spurious
        peaks)
    DEBUG : bool
        Test code used during development.  Should DEFINITELY be removed.

    Returns
    -------
    dx,dy : float,float
        REVERSE of dftregistration order (also, signs flipped) for consistency
        with other routines.
        Measures the amount im2 is offset from im1 (i.e., shift im2 by these #'s
        to match im1)

    """
    if not im1.shape == im2.shape:
        raise ValueError("Images must have same shape.")

    if zeromean:
        im1 = im1 - (im1[im1==im1].mean())
        im2 = im2 - (im2[im2==im2].mean())

    if np.any(np.isnan(im1)):
        im1 = im1.copy()
        im1[im1!=im1] = 0
    if np.any(np.isnan(im2)):
        im2 = im2.copy()
        im2[im2!=im2] = 0

    fft2,ifft2 = fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    im1fft = fft2(im1)
    im2fft = fft2(im2)

    output = dftregistration(im1fft,im2fft,usfac=usfac,
            return_registered=return_registered, return_error=return_error,
            zeromean=zeromean, DEBUG=DEBUG, maxoff=maxoff)

    output = [-output[1], -output[0], ] + [o for o in output[2:]]

    if return_registered:
        output[-1] = np.abs(np.fft.ifftshift(ifft2(output[-1])))

    return output


def dftregistration(buf1ft,buf2ft,usfac=1, return_registered=False,
        return_error=False, zeromean=True, DEBUG=False, maxoff=None,
        nthreads=1, use_numpy_fft=False):
    """
    Efficient subpixel image registration by crosscorrelation. This code
    gives the same precision as the FFT upsampled cross correlation in a
    small fraction of the computation time and with reduced memory 
    requirements. It obtains an initial estimate of the crosscorrelation peak
    by an FFT and then refines the shift estimation by upsampling the DFT
    only in a small neighborhood of that estimate by means of a 
    matrix-multiply DFT. With this procedure all the image points are used to
    compute the upsampled crosscorrelation.
    Manuel Guizar - Dec 13, 2007

    Portions of this code were taken from code written by Ann M. Kowalczyk 
    and James R. Fienup. 
    J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued 
    object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458 
    (1990).

    Citation for this algorithm:
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
    "Efficient subpixel image registration algorithms," Opt. Lett. 33, 
    156-158 (2008).

    Inputs
    buf1ft    Fourier transform of reference image, 
           DC in (1,1)   [DO NOT FFTSHIFT]
    buf2ft    Fourier transform of image to register, 
           DC in (1,1) [DO NOT FFTSHIFT]
    usfac     Upsampling factor (integer). Images will be registered to 
           within 1/usfac of a pixel. For example usfac = 20 means the
           images will be registered within 1/20 of a pixel. (default = 1)

    Outputs
    output =  [error,diffphase,net_row_shift,net_col_shift]
    error     Translation invariant normalized RMS error between f and g
    diffphase     Global phase difference between the two images (should be
               zero if images are non-negative).
    net_row_shift net_col_shift   Pixel shifts between images
    Greg      (Optional) Fourier transform of registered version of buf2ft,
           the global phase difference is compensated for.
    """

    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab
    from numpy import conj,abs,arctan2,sqrt,real,imag,shape,zeros,trunc,ceil,floor,fix
    from numpy.fft import fftshift,ifftshift
    fft2,ifft2 = fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    # Compute error for no pixel shift
    if usfac == 0:
        raise ValueError("Upsample Factor must be >= 1")
        CCmax = sum(sum(buf1ft * conj(buf2ft))); 
        rfzero = sum(abs(buf1ft)**2);
        rgzero = sum(abs(buf2ft)**2); 
        error = 1.0 - CCmax * conj(CCmax)/(rgzero*rfzero); 
        error = sqrt(abs(error));
        diffphase=arctan2(imag(CCmax),real(CCmax)); 
        output=[error,diffphase];
            
    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
    # peak
    elif usfac == 1:
        [m,n]=shape(buf1ft);
        CC = ifft2(buf1ft * conj(buf2ft));
        if maxoff is None:
            rloc,cloc = np.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]; 
        else:
            # set the interior of the shifted array to zero
            # (i.e., ignore it)
            CC[maxoff:-maxoff,:] = 0
            CC[:,maxoff:-maxoff] = 0
            rloc,cloc = np.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]; 
        rfzero = sum(abs(buf1ft)**2)/(m*n);
        rgzero = sum(abs(buf2ft)**2)/(m*n); 
        error = 1.0 - CCmax * conj(CCmax)/(rgzero*rfzero);
        error = sqrt(abs(error));
        diffphase=arctan2(imag(CCmax),real(CCmax)); 
        md2 = fix(m/2); 
        nd2 = fix(n/2);
        if rloc > md2:
            row_shift = rloc - m;
        else:
            row_shift = rloc;

        if cloc > nd2:
            col_shift = cloc - n;
        else:
            col_shift = cloc;
        #output=[error,diffphase,row_shift,col_shift];
        output=[row_shift,col_shift]
        
    # Partial-pixel shift
    else:
        
        if DEBUG: import pylab
        # First upsample by a factor of 2 to obtain initial estimate
        # Embed Fourier data in a 2x larger array
        [m,n]=shape(buf1ft);
        mlarge=m*2;
        nlarge=n*2;
        CClarge=zeros([mlarge,nlarge], dtype='complex');
        #CClarge[m-fix(m/2):m+fix((m-1)/2)+1,n-fix(n/2):n+fix((n-1)/2)+1] = fftshift(buf1ft) * conj(fftshift(buf2ft));
        CClarge[round(mlarge/4.):round(mlarge/4.*3),round(nlarge/4.):round(nlarge/4.*3)] = fftshift(buf1ft) * conj(fftshift(buf2ft));
        # note that matlab uses fix which is trunc... ?
      
        # Compute crosscorrelation and locate the peak 
        CC = ifft2(ifftshift(CClarge)); # Calculate cross-correlation
        if maxoff is None:
            rloc,cloc = np.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]; 
        else:
            # set the interior of the shifted array to zero
            # (i.e., ignore it)
            CC[maxoff:-maxoff,:] = 0
            CC[:,maxoff:-maxoff] = 0
            rloc,cloc = np.unravel_index(abs(CC).argmax(), CC.shape)
            CCmax=CC[rloc,cloc]; 

        if DEBUG:
            pylab.figure(1)
            pylab.clf()
            pylab.subplot(131)
            pylab.imshow(real(CC)); pylab.title("Cross-Correlation (upsampled 2x)")
            pylab.subplot(132)
            ups = dftups((buf1ft) * conj((buf2ft)),mlarge,nlarge,2,0,0); pylab.title("dftups upsampled 2x")
            pylab.imshow(real(((ups))))
            pylab.subplot(133)
            pylab.imshow(real(CC)/real(ups)); pylab.title("Ratio upsampled/dftupsampled")
            print "Upsample by 2 peak: ",rloc,cloc," using dft version: ",np.unravel_index(abs(ups).argmax(), ups.shape)
            #print np.unravel_index(ups.argmax(),ups.shape)
        
        # Obtain shift in original pixel grid from the position of the
        # crosscorrelation peak 
        [m,n] = shape(CC); md2 = trunc(m/2); nd2 = trunc(n/2);
        if rloc > md2 :
            row_shift2 = rloc - m;
        else:
            row_shift2 = rloc;
        if cloc > nd2:
            col_shift2 = cloc - n;
        else:
            col_shift2 = cloc;
        row_shift2=row_shift2/2.;
        col_shift2=col_shift2/2.;
        if DEBUG: print "row_shift/col_shift from ups2: ",row_shift2,col_shift2

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            #%% DFT computation %%%
            # Initial shift estimate in upsampled grid
            zoom_factor=1.5
            if DEBUG: print row_shift2, col_shift2
            row_shift0 = round(row_shift2*usfac)/usfac; 
            col_shift0 = round(col_shift2*usfac)/usfac;     
            dftshift = trunc(ceil(usfac*zoom_factor)/2); #% Center of output array at dftshift+1
            if DEBUG: print 'dftshift,rs,cs,zf:',dftshift, row_shift0, col_shift0, usfac*zoom_factor
            # Matrix multiply DFT around the current shift estimate
            roff = dftshift-row_shift0*usfac
            coff = dftshift-col_shift0*usfac
            upsampled = dftups(
                    (buf2ft * conj(buf1ft)),
                    ceil(usfac*zoom_factor),
                    ceil(usfac*zoom_factor), 
                    usfac, 
                    roff,
                    coff)
            #CC = conj(dftups(buf2ft.*conj(buf1ft),ceil(usfac*1.5),ceil(usfac*1.5),usfac,...
            #    dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac^2);
            CC = conj(upsampled)/(md2*nd2*usfac**2);
            if DEBUG:
                pylab.figure(2)
                pylab.clf()
                pylab.subplot(221)
                pylab.imshow(abs(upsampled)); pylab.title('upsampled')
                pylab.subplot(222)
                pylab.imshow(abs(CC)); pylab.title('CC upsampled')
                pylab.subplot(223); pylab.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(buf2ft * conj(buf1ft))))); pylab.title('xc')
                yy,xx = np.indices([m*usfac,n*usfac],dtype='float')
                pylab.contour(yy/usfac/2.-0.5+1,xx/usfac/2.-0.5-1, np.abs(dftups((buf2ft*conj(buf1ft)),m*usfac,n*usfac,usfac)))
                pylab.subplot(224); pylab.imshow(np.abs(dftups((buf2ft*conj(buf1ft)),ceil(usfac*zoom_factor),ceil(usfac*zoom_factor),usfac))); pylab.title('unshifted ups')
            # Locate maximum and map back to original pixel grid 
            rloc,cloc = np.unravel_index(abs(CC).argmax(), CC.shape) 
            rloc0,cloc0 = np.unravel_index(abs(CC).argmax(), CC.shape) 
            CCmax = CC[rloc,cloc]
            #[max1,loc1] = CC.max(axis=0), CC.argmax(axis=0)
            #[max2,loc2] = max1.max(),max1.argmax()
            #rloc=loc1[loc2];
            #cloc=loc2;
            #CCmax = CC[rloc,cloc];
            rg00 = dftups(buf1ft * conj(buf1ft),1,1,usfac)/(md2*nd2*usfac**2);
            rf00 = dftups(buf2ft * conj(buf2ft),1,1,usfac)/(md2*nd2*usfac**2);  
            #if DEBUG: print rloc,row_shift,cloc,col_shift,dftshift
            rloc = rloc - dftshift #+ 1 # +1 # questionable/failed hack + 1;
            cloc = cloc - dftshift #+ 1 # -1 # questionable/failed hack - 1;
            #if DEBUG: print rloc,row_shift,cloc,col_shift,dftshift
            row_shift = row_shift0 + rloc/usfac;
            col_shift = col_shift0 + cloc/usfac;    
            #if DEBUG: print rloc/usfac,row_shift,cloc/usfac,col_shift
            if DEBUG: print "Off by: ",(0.25 - float(rloc)/usfac)*usfac , (-0.25 - float(cloc)/usfac)*usfac 
            if DEBUG: print "correction was: ",rloc/usfac, cloc/usfac
            if DEBUG: print "Coordinate went from",row_shift2,col_shift2,"to",row_shift0,col_shift0,"to", row_shift, col_shift
            if DEBUG: print "dftsh - usfac:", dftshift-usfac
            if DEBUG: print  rloc,cloc,row_shift,col_shift,CCmax,dftshift,rloc0,cloc0

        # If upsampling = 2, no additional pixel shift refinement
        else:    
            rg00 = sum(sum( buf1ft * conj(buf1ft) ))/m/n;
            rf00 = sum(sum( buf2ft * conj(buf2ft) ))/m/n;
            row_shift = row_shift2
            col_shift = col_shift2
        error = 1.0 - CCmax * conj(CCmax)/(rg00*rf00);
        error = sqrt(abs(error));
        diffphase=arctan2(imag(CCmax),real(CCmax));
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        if md2 == 1:
            row_shift = 0;
        if nd2 == 1:
            col_shift = 0;
        #output=[error,diffphase,row_shift,col_shift];
        output=[row_shift,col_shift]

    if return_error:
        # simple estimate of the precision of the fft approach
        output += [1./usfac,1./usfac]

    # Compute registered version of buf2ft
    if (return_registered):
        if (usfac > 0):
            nr,nc=shape(buf2ft);
            Nr = np.fft.ifftshift(np.linspace(-np.fix(nr/2),np.ceil(nr/2)-1,nr))
            Nc = np.fft.ifftshift(np.linspace(-np.fix(nc/2),np.ceil(nc/2)-1,nc))
            [Nc,Nr] = meshgrid(Nc,Nr);
            Greg = buf2ft * exp(1j*2*pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
            Greg = Greg*exp(1j*diffphase);
        elif (usfac == 0):
            Greg = buf2ft*exp(1j*diffphase);
        output.append(Greg)

    return output

def dftups(inp,nor=None,noc=None,usfac=1,roff=0,coff=0):
    """
    Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
    a small region.
    usfac         Upsampling factor (default usfac = 1)
    [nor,noc]     Number of pixels in the output upsampled DFT, in
                  units of upsampled pixels (default = size(in))
    roff, coff    Row and column offsets, allow to shift the output array to
                  a region of interest on the DFT (default = 0)
    Recieves DC in upper left corner, image center must be in (1,1) 
    Manuel Guizar - Dec 13, 2007
    Modified from dftus, by J.R. Fienup 7/31/06

    This code is intended to provide the same result as if the following
    operations were performed
      - Embed the array "in" in an array that is usfac times larger in each
        dimension. ifftshift to bring the center of the image to (1,1).
      - Take the FFT of the larger array
      - Extract an [nor, noc] region of the result. Starting with the 
        [roff+1 coff+1] element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the
    zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
    """
    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab
    from numpy.fft import ifftshift
    from numpy import pi,newaxis,floor

    nr,nc=np.shape(inp);
    # Set defaults
    if noc is None: noc=nc;
    if nor is None: nor=nr;
    # Compute kernels and obtain DFT by matrix products
    kernc=np.exp((-1j*2*pi/(nc*usfac))*( ifftshift(np.arange(nc) - floor(nc/2)).T[:,newaxis] )*( np.arange(noc) - coff  )[newaxis,:]);
    kernr=np.exp((-1j*2*pi/(nr*usfac))*( np.arange(nor).T - roff )[:,newaxis]*( ifftshift(np.arange(nr)) - floor(nr/2) )[newaxis,:]);
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    out=np.dot(np.dot(kernr,inp),kernc);
    #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
    return out 

def upsample_image(image, upsample_factor=1, output_size=None, nthreads=1, use_numpy_fft=False,
        xshift=0, yshift=0):
    """
    Use dftups to upsample an image (but takes an image and returns an image with all reals)
    """
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    imfft = ifftn(image)

    if output_size is None:
        s1 = image.shape[0]*upsample_factor
        s2 = image.shape[1]*upsample_factor
    elif hasattr(output_size,'__len__'):
        s1 = output_size[0]
        s2 = output_size[1]
    else:
        s1 = output_size
        s2 = output_size

    ups = dftups(imfft, s1, s2, upsample_factor, roff=yshift, coff=xshift)

    return np.abs(ups)



    

def upsample_ft_raw(buf1ft,buf2ft,zoomfac=2):
    """
    This was just test/debug code to compare to dftups; it is not meant for use
    """

    from numpy.fft import ifft2,ifftshift,fftshift
    from numpy import conj

    [m,n]=np.shape(buf1ft);
    mlarge=m*zoomfac;
    nlarge=n*zoomfac;
    CClarge=np.zeros([mlarge,nlarge], dtype='complex');
    #CClarge[m-fix(m/2):m+fix((m-1)/2)+1,n-fix(n/2):n+fix((n-1)/2)+1] = fftshift(buf1ft) * conj(fftshift(buf2ft));
    #CClarge[mlarge/4:mlarge/4*3,nlarge/4:nlarge/4*3] = fftshift(buf1ft) * conj(fftshift(buf2ft));
    CClarge[round(mlarge/(zoomfac*2.)):round(mlarge/(zoomfac*2.)*3),round(nlarge/(zoomfac*2.)):round(nlarge/(zoomfac*2.)*3)] = fftshift(buf1ft) * conj(fftshift(buf2ft));
        # note that matlab uses fix which is trunc... ?
  
    # Compute crosscorrelation and locate the peak 
    CC = ifft2(ifftshift(CClarge)); # Calculate cross-correlation
    
    return CC

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
    

def chi2_of_offset(im1, im2, err=None, upsample_factor=10, boundary='wrap',
        nthreads=1, use_numpy_fft=False, zeromean=False, ndof=2, verbose=True,
        return_error=True, return_chi2array=False):
    """
    Find the offsets between image 1 and image 2 using the DFT upsampling method
    (http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html)
    combined with chi^2 to measure the errors on the fit
    
    Parameters
    ----------
    im1 : np.ndarray
    im2 : np.ndarray
        The images to register. 
    err : np.ndarray
        Per-pixel error in image 2
    boundary : 'wrap','constant','reflect','nearest'
        Option to pass to map_coordinates for determining what to do with
        shifts outside of the boundaries.  
    upsample_factor : int
        upsampling factor; governs accuracy of fit (1/usfac is best accuracy)
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
    ndof : int
        number of degrees of freedom in the fit (used for chi^2 computations).
        Should probably always be 2.


    Returns
    -------
    dx,dy : float,float
        REVERSE of dftregistration order (also, signs flipped) for consistency
        with other routines.
        Measures the amount im2 is offset from im1 (i.e., shift im2 by these #'s
        to match im1)


    """
    if not im1.shape == im2.shape:
        raise ValueError("Images must have same shape.")

    if zeromean:
        im1 = im1 - (im1[im1==im1].mean())
        im2 = im2 - (im2[im2==im2].mean())

    if np.any(np.isnan(im1)):
        im1 = im1.copy()
        im1[im1!=im1] = 0
    if np.any(np.isnan(im2)):
        im2 = im2.copy()
        im2[im2!=im2] = 0

    xc = correlate2d(im1,im2, boundary=boundary)
    if err is not None:
        err_ac = correlate2d(err,err, boundary=boundary)
        err2sum = (err**2).sum()
    else:
        err_ac = xc.size - ndof
        err2sum = xc.size - ndof
    ac1peak = (im1**2).sum()
    ac2peak = (im2**2).sum()
    chi2n = (ac1peak/err2sum - 2*xc/err_ac + ac2peak/err2sum)
    ymax, xmax = np.unravel_index(chi2n.argmin(), chi2n.shape)

    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    # biggest scale = where chi^2/n ~ 9?

    s1,s2 = im1.shape
    ylen,xlen = im1.shape
    xcen = xlen/2-(1-xlen%2) 
    ycen = ylen/2-(1-ylen%2) 
    yshift = ymax-ycen # shift im2 by these numbers to get im1
    xshift = xmax-xcen

    #xcft = correlate2d(im1,im2,boundary=boundary,return_fft=True)

    # signs?!
    zoom_factor = s1/upsample_factor
    if zoom_factor <= 1:
        zoom_factor = 2
        s1 = zoom_factor*upsample_factor
        s2 = zoom_factor*upsample_factor
    dftshift = np.trunc(np.ceil(upsample_factor*zoom_factor)/2); #% Center of output array at dftshift+1
    xc_ups = dftups(fftn(im2)*np.conj(fftn(im1)), s1, s2, usfac=upsample_factor,
            roff=dftshift-yshift*upsample_factor,
            coff=dftshift-xshift*upsample_factor) / (im1.size) #*upsample_factor**2)
    if err is not None:
        err_ups = upsample_image(err, output_size=s1,
                upsample_factor=upsample_factor, xshift=xshift, yshift=yshift)
    else:
        err_ups = 1
    chi2n_ups = (ac1peak/err2sum-2*np.abs(xc_ups)/np.abs(err_ups)+ac2peak/err2sum)
    deltachi2 = chi2n_ups - chi2n_ups.min()

    yy,xx = np.indices([s1,s2])
    xshifts_corrections = (xx-dftshift)/upsample_factor
    yshifts_corrections = (yy-dftshift)/upsample_factor

    try:
        import scipy.stats
        # 1,2,3-sigma delta-chi2 levels
        m1 = scipy.stats.chi2.ppf( 1-scipy.stats.norm.sf(1)*2, ndof )
        m2 = scipy.stats.chi2.ppf( 1-scipy.stats.norm.sf(2)*2, ndof )
        m3 = scipy.stats.chi2.ppf( 1-scipy.stats.norm.sf(3)*2, ndof )
    except ImportError:
        # assume m=2 (2 degrees of freedom)
        m1 = 2.2957489288986364
        m2 = 6.1800743062441734 
        m3 = 11.829158081900793

    sigma1_area = deltachi2<m1
    x_sigma1 = xshifts_corrections[sigma1_area]
    y_sigma1 = yshifts_corrections[sigma1_area]
    # optional...?
    #sigma2_area = deltachi2<m2
    #x_sigma2 = xshifts_corrections[sigma2_area]
    #y_sigma2 = yshifts_corrections[sigma2_area]
    #sigma3_area = deltachi2<m3
    #x_sigma3 = xshifts_corrections[sigma3_area]
    #y_sigma3 = yshifts_corrections[sigma3_area]

    if sigma1_area.sum() <= 1:
        if verbose:
            print "Cannot estimate errors: need higher upsample factor"
        errx_low = erry_low = errx_high = erry_high = 1./upsample_factor

    upsymax,upsxmax = np.unravel_index(chi2n_ups.argmin(), chi2n_ups.shape)

    errx_low = (upsxmax-dftshift)/upsample_factor - x_sigma1.min()
    errx_high = x_sigma1.max() - (upsxmax-dftshift)/upsample_factor
    erry_low = (upsymax-dftshift)/upsample_factor - y_sigma1.min()
    erry_high = y_sigma1.max() - (upsymax-dftshift)/upsample_factor

    yshift_corr = yshift+(upsymax-dftshift)/float(upsample_factor)
    xshift_corr = xshift+(upsxmax-dftshift)/float(upsample_factor)
    if verbose > 1:
        #print ymax,xmax
        #print upsymax, upsxmax
        #print upsymax-dftshift, upsxmax-dftshift
        print "Correction: ",(upsymax-dftshift)/float(upsample_factor), (upsxmax-dftshift)/float(upsample_factor)
        print "Chi2 1sig bounds:", x_sigma1.min(), x_sigma1.max(), y_sigma1.min(), y_sigma1.max()
        print errx_low,errx_high,erry_low,erry_high
        print "%0.3f +%0.3f -%0.3f   %0.3f +%0.3f -%0.3f" % (yshift_corr, erry_high, erry_low, xshift_corr, errx_high, errx_low)
        #print ymax-ycen+upsymax/float(upsample_factor), xmax-xcen+upsxmax/float(upsample_factor)
        #print (upsymax-s1/2)/upsample_factor, (upsxmax-s2/2)/upsample_factor

    shift_xvals = xshifts_corrections+xshift
    shift_yvals = yshifts_corrections+yshift

    returns = [yshift_corr,xshift_corr]
    if return_error:
        returns.append( (errx_low+errx_high)/2. )
        returns.append( (erry_low+erry_high)/2. )
    if return_chi2array:
        returns.append((shift_xvals,shift_yvals,chi2n_ups))

    return returns


def chi2_shift(im1, im2, err=None, mode='wrap', maxoff=None, return_error=True,
        guessx=0, guessy=0, use_fft=False, ignore_outside=True, **kwargs):
    """
    Determine the best fit offset using `scipy.ndimage.map_coordinates` to
    shift the offset image.

    Parameters
    ----------
        im1 : np.ndarray
            First image
        im2 : np.ndarray
            Second image (offset image)
        err : np.ndarray
            Per-pixel error in image 2
        mode : 'wrap','constant','reflect','nearest'
            Option to pass to map_coordinates for determining what to do with
            shifts outside of the boundaries.  
        maxoff : None or int
            If set, crop the data after shifting before determining chi2
            (this is a good thing to use; not using it can result in weirdness
            involving the boundaries)

    """
    #xc = correlate2d(im1,im2, boundary=boundary)
    #ac1peak = (im1**2).sum()
    #ac2peak = (im2**2).sum()
    #chi2 = ac1peak - 2*xc + ac2peak


    if not im1.shape == im2.shape:
        raise ValueError("Images must have same shape.")

    if np.any(np.isnan(im1)):
        im1 = im1.copy()
        im1[im1!=im1] = 0
    if np.any(np.isnan(im2)):
        im2 = im2.copy()
        if err is not None:
            err[im2!=im2] = np.inf
        im2[im2!=im2] = 0

    im1 = im1-im1.mean()
    im2 = im2-im2.mean()
    if not use_fft:
        yy,xx = np.indices(im1.shape)
    ylen,xlen = im1.shape
    xcen = xlen/2-(1-xlen%2) 
    ycen = ylen/2-(1-ylen%2) 
    import scipy.ndimage,scipy.optimize

    def residuals(p, **kwargs):
        xsh,ysh = p
        if use_fft:
            shifted_img = shift(im2, -ysh, -xsh)
        else:
            shifted_img = scipy.ndimage.map_coordinates(im2, [yy+ysh,xx+xsh], mode=mode, **kwargs)
        if maxoff is not None:
            xslice = slice(xcen-maxoff,xcen+maxoff,None)
            yslice = slice(ycen-maxoff,ycen+maxoff,None)
            # divide by sqrt(number of samples) = sqrt(maxoff**2)
            residuals = np.abs(np.ravel((im1[yslice,xslice]-shifted_img[yslice,xslice])) / maxoff)
        else:
            if ignore_outside:
                outsidex = min([(xlen-2*xsh)/2,xcen])
                outsidey = min([(ylen-2*ysh)/2,xcen])
                xslice = slice(xcen-outsidex,xcen+outsidex,None)
                yslice = slice(ycen-outsidey,ycen+outsidey,None)
                residuals = ( np.abs( np.ravel(
                    (im1[yslice,xslice]-shifted_img[yslice,xslice]))) /
                    (2*outsidex*2*outsidey)**0.5 )
            else:
                xslice = slice(None)
                yslice = slice(None)
                residuals = np.abs(np.ravel((im1-shifted_img))) / im1.size**0.5
        if err is None:
            return residuals
        else:
            if use_fft:
                shifted_err = shift(err, -ysh, -xsh)
            else:
                shifted_err = scipy.ndimage.map_coordinates(err, [yy+ysh,xx+xsh], mode=mode, **kwargs)
            return residuals / shifted_err[yslice,xslice].flat

    bestfit,cov,info,msg,ier = scipy.optimize.leastsq(residuals, [guessx,guessy], full_output=1)
    #bestfit,cov = scipy.optimize.curve_fit(shift, im2, im1, p0=[guessx,guessy], sigma=err)

    chi2n = (residuals(bestfit)**2).sum() / (im1.size-2)

    if bestfit is None or cov is None:
        print bestfit-np.array([guessx,guessy])
        print bestfit
        print cov
        print info
        print msg
        print ier
        if cov is None:
            from numpy.dual import inv
            from numpy.linalg import LinAlgError
            n = 2 # number of free pars
            perm = np.take(np.eye(n),info['ipvt']-1,0)
            r = np.triu(np.transpose(info['fjac'])[:n,:])
            R = np.dot(r, perm)
            try:
                cov = inv(np.dot(np.transpose(R),R))
            except LinAlgError:
                print "Could not compute cov because of linalgerr"
                pass
                
        
    if return_error:
        if cov is None:
            return bestfit[0],bestfit[1],0,0
        else: # based on scipy.optimize.curve_fit, the "correct" covariance is this cov * chi^2/n
            return bestfit[0],bestfit[1],(cov[0,0]*chi2n)**0.5,(cov[1,1]*chi2n)**0.5
    else:
        return bestfit[0],bestfit[1]



try:
    import pytest
    import itertools
    from scipy import interpolate

    shifts = [1,1.5,-1.25,8.2,10.1]
    sizes = [99,100,101]
    amps = [5.,10.,50.,100.,500.,1000.]
    gaussfits = (True,False)

    def make_offset_images(xsh,ysh,imsize, width=3.0, amp=1000.0, noiseamp=1.0,
            xcen=50, ycen=50):
        image = np.random.randn(imsize,imsize) * noiseamp
        Y, X = np.indices([imsize, imsize])
        X -= xcen
        Y -= ycen
        new_r = np.sqrt(X*X+Y*Y)
        image += amp*np.exp(-(new_r)**2/(2.*width**2))

        tolerance = 3. * 1./np.sqrt(2*np.pi*width**2*amp/noiseamp)

        new_image = np.random.randn(imsize,imsize)*noiseamp + amp*np.exp(-((X-xsh)**2+(Y-ysh)**2)/(2.*width**2))

        return image, new_image, tolerance

    def make_extended(imsize, powerlaw=2.0):
        yy,xx = np.indices((imsize,imsize))
        cen = imsize/2-(1-imsize%2) 
        yy -= cen
        xx -= cen
        rr = (xx**2+yy**2)**0.5
        
        powermap = (np.random.randn(imsize,imsize) * rr**(-powerlaw)+
            np.random.randn(imsize,imsize) * rr**(-powerlaw) * 1j)
        powermap[powermap!=powermap] = 0

        newmap = np.abs(np.fft.fftshift(np.fft.fft2(powermap)))

        return newmap

    def make_offset_extended(img, xsh, ysh, noise=1.0, mode='wrap'):
        import scipy, scipy.ndimage
        #yy,xx = np.indices(img.shape,dtype='float')
        #yy-=ysh
        #xx-=xsh
        noise = np.random.randn(*img.shape)*noise
        #newimage = scipy.ndimage.map_coordinates(img+noise, [yy,xx], mode=mode)
        newimage = np.abs(shift(img+noise, xsh, ysh))

        return newimage



    @pytest.mark.parametrize(('xsh','ysh','imsize','gaussfit'),list(itertools.product(shifts,shifts,sizes,gaussfits)))
    def test_shifts(xsh,ysh,imsize,gaussfit):
        image,new_image,tolerance = make_offset_images(xsh, ysh, imsize)
        if gaussfit:
            xoff,yoff,exoff,eyoff = cross_correlation_shifts(image,new_image)
            print xoff,yoff,np.abs(xoff-xsh),np.abs(yoff-ysh),exoff,eyoff
        else:
            xoff,yoff = cross_correlation_shifts(image,new_image)
            print xoff,yoff,np.abs(xoff-xsh),np.abs(yoff-ysh) 
        assert np.abs(xoff-xsh) < tolerance
        assert np.abs(yoff-ysh) < tolerance

    def do_n_fits(nfits, xsh, ysh, imsize, gaussfit=False, maxoff=None,
            return_error=False, shift_func=cross_correlation_shifts,
            sfkwargs={}, **kwargs):
        """
        Test code

        Parameters
        ----------
        nfits : int
            Number of times to perform fits
        xsh : float
            X shift from input to output image
        ysh : float
            Y shift from input to output image
        imsize : int
            Size of image (square)
        """
        offsets = [
            shift_func( 
                *make_offset_images(xsh, ysh, imsize, **kwargs)[:2],
                gaussfit=gaussfit, maxoff=maxoff, return_error=return_error)
            for ii in xrange(nfits)]

        return offsets

    def do_n_fits_register(nfits, xsh, ysh, imsize, usfac=10,
            return_error=False, **kwargs):
        """
        Test code

        Parameters
        ----------
        nfits : int
            Number of times to perform fits
        xsh : float
            X shift from input to output image
        ysh : float
            Y shift from input to output image
        imsize : int
            Size of image (square)
        """
        offsets = []
        for ii in xrange(nfits):
            im1,im2,temp = make_offset_images(xsh, ysh, imsize, **kwargs)
            xoff,yoff,reg = register(
                im1, im2,
                return_registered=True, usfac=usfac)
            chi2 = ((im1-reg)**2).sum() / im1.size
            offsets.append([xoff,yoff,chi2])

        return offsets

    def do_n_extended_fits(nfits, xsh, ysh, imsize,  gaussfit=False,
            maxoff=None, return_error=False, powerlaw=2.0, noise=1.0,
            unsharp_mask=False, smoothfactor=5, zeropad=0,
            shift_func=cross_correlation_shifts, sfkwargs={},
            doplot=False,
            **kwargs):

        try: 
            import progressbar
            widgets = [progressbar.FormatLabel('Processed: %(value)d offsets in %(elapsed)s)'), progressbar.Percentage()]
            progress = progressbar.ProgressBar(widgets=widgets)
        except ImportError:
            def progress(x):
                yield x

        image = make_extended(imsize, powerlaw=powerlaw)
        if zeropad > 0:
            newsize = [s+zeropad for s in image.shape]
            ylen,xlen = newsize
            xcen = xlen/2-(1-xlen%2) 
            ycen = ylen/2-(1-ylen%2) 
            newim = np.zeros(newsize)
            newim[ycen-image.shape[0]/2:ycen+image.shape[0]/2, xcen-image.shape[1]/2:xcen+image.shape[1]/2] = image
            image = newim


        if unsharp_mask:
            from AG_fft_tools import smooth
            offsets = []
            for ii in progress(xrange(nfits)):
                inim = image-smooth(image,smoothfactor)
                offim = make_offset_extended(image, xsh, ysh, noise=noise, **kwargs)
                offim -= smooth(offim,smoothfactor)
                offsets.append( shift_func( inim, offim,  return_error=return_error, **sfkwargs) )
        else:
            offsets = []
            if doplot:
                import pylab
                pylab.figure(3); pylab.subplot(221); pylab.imshow(image-image.mean()); pylab.subplot(222); pylab.imshow(offim-offim.mean())
                #subplot(223); pylab.imshow((abs(fft2(image-image.mean())*conj(fft2(offim-offim.mean())))))
                pylab.subplot(223); pylab.imshow(abs(ifft2((fft2(image)*conj(fft2(offim))))))
                pylab.subplot(224); pylab.imshow(abs(ifft2((fft2(image-image.mean())*conj(fft2(offim-offim.mean()))))))
                draw()
            for ii in progress(xrange(nfits)):
                offim = make_offset_extended(image, xsh, ysh, noise=noise, **kwargs)
                offsets.append( shift_func( 
                    image,
                    offim,
                    return_error=return_error, **sfkwargs)
                    )

        return offsets


    #@pytest.mark.parametrize(('xsh','ysh','imsize','amp','gaussfit'),
    #        list(itertools.product(shifts,shifts,sizes,amps,gaussfits)))
    def run_tests(xsh, ysh, imsize, amp, gaussfit, nfits=1000, maxoff=20):
        fitted_shifts = np.array(do_n_fits(nfits, xsh, ysh, imsize, amp=amp, maxoff=maxoff))
        errors = fitted_shifts.std(axis=0)
        x,y,ex,ey = cross_correlation_shifts(
                *make_offset_images(xsh, ysh, imsize, amp=amp)[:2],
                gaussfit=gaussfit, maxoff=maxoff, return_error=True,
                errim1=np.ones([imsize,imsize]),
                errim2=np.ones([imsize,imsize]))
        print "StdDev: %10.3g,%10.3g  Measured: %10.3g,%10.3g "+\
                " Difference: %10.3g, %10.3g  Diff/Real: %10.3g,%10.3g" % (
                errors[0],errors[1], ex,ey,errors[0]-ex,errors[1]-ey,
                (errors[0]-ex)/errors[0], (errors[1]-ey)/errors[1])

        return errors[0],errors[1],ex,ey


    def plot_tests(nfits=25,xsh=1.75,ysh=1.75, imsize=64, amp=10., **kwargs):
        x,y,ex,ey = np.array(do_n_fits(nfits, xsh, ysh, imsize, amp,
            maxoff=12., return_error=True, **kwargs)).T
        import pylab
        pylab.plot([xsh],[ysh],'kd',markersize=20)
        pylab.errorbar(x,y,xerr=ex,yerr=ey,linestyle='none')

    def plot_extended_tests(nfits=25,xsh=1.75,ysh=1.75, imsize=64, noise=1.0,
            maxoff=12., zeropad=64, **kwargs):
        x,y,ex,ey = np.array(do_n_extended_fits(nfits, xsh, ysh, imsize, 
            maxoff=maxoff, return_error=True, noise=noise, **kwargs)).T
        print x,y
        import pylab
        pylab.plot([xsh],[ysh],'kd',markersize=20)
        pylab.errorbar(x,y,xerr=ex,yerr=ey,linestyle='none')

    def determine_error_offsets():
        """
        Experiment to determine how wrong the error estimates are
        (WHY are they wrong?  Still don't understand)
        """
        # analytic
        A = np.array([run_tests(1.5,1.5,50,a,False,nfits=200) for a in np.logspace(1.5,3,30)]);
        G = np.array([run_tests(1.5,1.5,50,a,True,nfits=200) for a in np.logspace(1.5,3,30)]);
        print "Analytic offset: %g" % (( (A[:,3]/A[:,1]).mean() + (A[:,2]/A[:,0]).mean() )/2. )
        print "Gaussian offset: %g" % (( (G[:,3]/G[:,1]).mean() + (G[:,2]/G[:,0]).mean() )/2. )
        
    def test_upsample(imsize, usfac=2, xsh=2.25, ysh=2.25, noise=0.1, **kwargs):
        image = make_extended(imsize)
        offim = make_offset_extended(image, xsh, ysh, noise=noise, **kwargs)

        raw_us = upsample_ft_raw(np.fft.fft2(image-image.mean()), np.fft.fft2(offim-offim.mean()), zoomfac=usfac)
        dftus = dftups(np.fft.fft2(image-image.mean())*np.conj( np.fft.fft2(offim-offim.mean())), imsize*usfac, imsize*usfac, usfac, 0, 0)

        import pylab
        pylab.clf()
        pylab.subplot(221); pylab.imshow(abs(raw_us))
        pylab.subplot(222); pylab.imshow(abs(dftus[::-1,::-1]))
        pylab.subplot(223); pylab.imshow(abs(dftus[::-1,::-1]/dftus.max()-raw_us/raw_us.max()))
        pylab.subplot(224); pylab.imshow(abs(dftus[::-1,::-1])); pylab.contour(raw_us)

    def accuracy_plot(xsh=2.25,ysh=-1.35,amp=10000,width=1,imsize=100,usf_range=[1,100]):
        testg,testgsh,T = make_offset_images(xsh,ysh,imsize,amp=amp,width=width)
        offsets = []
        for usf in xrange(*usf_range): 
            dy,dx = dftregistration(np.fft.fft2(testg),np.fft.fft2(testgsh),usfac=usf); 
            # offsets are negative...
            offsets.append([xsh+dx,ysh+dy])

        import pylab
        dr = (np.array(offsets)**2).sum(axis=1)**0.5
        pylab.plot(np.arange(*usf_range), dr, label="A=%0.1g w=%0.1g" % (amp,width))
        pylab.plot(np.arange(*usf_range), 1./np.arange(*usf_range), 'k--', label="Theoretical")

    def accuracy_plot_extended(xsh=2.25,ysh=-1.35,noise=0.1,imsize=100,usf_range=[1,100]):
        offsets = []
        for usf in xrange(*usf_range): 
            dy,dx = do_n_extended_fits(1,xsh,ysh, imsize, shift_func=register,sfkwargs={'usfac':usf},noise=noise)[0]
            offsets.append([xsh+dx,ysh+dy])

        import pylab
        dr = (np.array(offsets)**2).sum(axis=1)**0.5
        pylab.plot(np.arange(*usf_range), dr, label="noise=%0.2g" % (noise))
        pylab.plot(np.arange(*usf_range), 1./np.arange(*usf_range), 'k--', label="Theoretical")

    def error_test(xsh=2.25,ysh=-1.35,noise=0.5,imsize=100,usf=101,nsamples=100,maxoff=10):
        """
        Empirically determine the error in the fit using random realizations, compare to...
        noise level, I guess?
        """

        offsets = np.array(do_n_extended_fits(nsamples, xsh, ysh, imsize,
            shift_func=register, sfkwargs={'usfac':usf,'maxoff':maxoff}, noise=noise))

        print "Theoretical accuracy: ",1./usf
        print "Standard Deviation x,y: ",offsets.std(axis=0)
        #print "Standard Deviation compared to truth x,y: ",(offsets-np.array([ysh,xsh])).std(axis=0)
        print "Mean x,y: ",offsets.mean(axis=0),"Real x,y: ",xsh,ysh
        print "Mean x,y - true x,y: ",offsets.mean(axis=0)-np.array([xsh,ysh])
        print "Mean x,y - true x,y / std: ",(offsets.mean(axis=0)-np.array([xsh,ysh]))/offsets.std(axis=0)
        signal = 3.05 * imsize**2 # empirical: plot(array([5,25,50,75,100,125,150]),array([mean([make_extended(jj).sum() for i in xrange(100)]) for jj in [5,25,50,75,100,125,150]])/array([5,25,50,75,100,125,150])**2)
        noise = 0.8 * imsize**2 * noise
        print "Signal / Noise: ", signal / noise

        return np.array(offsets),offsets.std(axis=0),offsets.mean(axis=0)+np.array([ysh,xsh]),signal/noise


    def register_accuracy_test(im1,im2,usf_range=[1,100],**kwargs):
        offsets = []
        try: 
            import progressbar
            widgets = [progressbar.FormatLabel('Processed: %(value)d offsets in %(elapsed)s)'), progressbar.Percentage()]
            progress = progressbar.ProgressBar(widgets=widgets)
        except ImportError:
            def progress(x):
                yield x
        for usf in progress(xrange(*usf_range)): 
            dy,dx = register(im1,im2,usfac=usf,**kwargs)
            offsets.append([dx,dy])

        return np.array(offsets)

    def register_noise_test(im1,im2, ntests=100, noise=np.std,
            register_method=register, **kwargs):
        """
        Perform tests with noise added to determine the errors on the 
        'best-fit' offset

        Parameters
        ----------
        register_method : function
            Which registration method to test
        ntests : int
            Number of tests to run
        noise : func or real
            Either a function to apply to im2 to determine the noise to use, or
            a fixed noise level
        """

        try:
            noise = noise(im2)
        except TypeError:
            pass

        try: 
            import progressbar
            widgets = [progressbar.FormatLabel('Processed: %(value)d offsets in %(elapsed)s'), progressbar.Percentage()]
            progress = progressbar.ProgressBar(widgets=widgets)
        except ImportError:
            def progress(x):
                yield x

        offsets = []
        for test_number in progress(xrange(ntests)):
            extra_noise = np.random.randn(*im2.shape) * noise
            dx,dy = register_method(im1,im2+extra_noise,**kwargs)
            offsets.append([dx,dy])

        return np.array(offsets)

    def compare_methods(im1,im2, ntests=100, noise=np.std,
            usfac=201, **kwargs):
        """
        Perform tests with noise added to determine the errors on the 
        'best-fit' offset

        Parameters
        ----------
        usfac : int
            upsampling factor; governs accuracy of fit (1/usfac is best accuracy)
        ntests : int
            Number of tests to run
        noise : func or real
            Either a function to apply to im2 to determine the noise to use, or
            a fixed noise level
        """

        try:
            noise = noise(im2)
        except TypeError:
            pass

        try: 
            import progressbar
            widgets = [progressbar.FormatLabel('Processed: %(value)d offsets in %(elapsed)s'), progressbar.Percentage()]
            progress = progressbar.ProgressBar(widgets=widgets)
        except ImportError:
            def progress(x):
                yield x

        offsets = []
        eoffsets = []
        for test_number in progress(xrange(ntests)):
            extra_noise = np.random.randn(*im2.shape) * noise
            dxr, dyr, edxr, edyr = register(im1, im2+extra_noise, usfac=usfac,
                    return_error=True, **kwargs)
            dxccs, dyccs, edxccs, edyccs = cross_correlation_shifts(im1,
                    im2+extra_noise, return_error=True, **kwargs)
            dxccg, dyccg, edxccg, edyccg = cross_correlation_shifts(im1,
                    im2+extra_noise, return_error=True, gaussfit=True,
                    **kwargs)
            dxchi, dychi, edxchi, edychi = chi2_of_offset(im1, im2+extra_noise,
                    return_error=True, **kwargs)
            offsets.append([dxr,dyr,dxccs,dyccs,dxccg,dyccg,dxchi,dychi])
            eoffsets.append([edxr,edyr,edxccs,edyccs,edxccg,edyccg,edxchi,edychi])

        return np.array(offsets),np.array(eoffsets)

    def plot_compare_methods(offsets, eoffsets, dx=None, dy=None, fig1=1,
            fig2=2, legend=True):
        """
        plot wrapper
        """
        import pylab

        pylab.figure(fig1)
        pylab.clf()
        if dx is not None and dy is not None:
            pylab.plot([dx],[dy],'kx',markersize=30,zorder=50,markeredgewidth=3)
        pylab.errorbar(offsets[:,0],offsets[:,1],xerr=eoffsets[:,0],yerr=eoffsets[:,1],linestyle='none',label='DFT')
        pylab.errorbar(offsets[:,2],offsets[:,3],xerr=eoffsets[:,2],yerr=eoffsets[:,3],linestyle='none',label='Taylor')
        pylab.errorbar(offsets[:,4],offsets[:,5],xerr=eoffsets[:,4],yerr=eoffsets[:,5],linestyle='none',label='Gaussian')
        pylab.errorbar(offsets[:,6],offsets[:,7],xerr=eoffsets[:,6],yerr=eoffsets[:,7],linestyle='none',label='$\\chi^2$')
        if legend:
            pylab.legend(loc='best')

        means = offsets.mean(axis=0)
        stds = offsets.std(axis=0)
        emeans = eoffsets.mean(axis=0)
        estds = eoffsets.std(axis=0)

        print stds
        print emeans
        print emeans/stds

        pylab.figure(fig2)
        pylab.clf()
        if dx is not None and dy is not None:
            pylab.plot([dx],[dy],'kx',markersize=30,zorder=50,markeredgewidth=3)
        pylab.errorbar(means[0],means[1],xerr=emeans[0],yerr=emeans[1],capsize=20,color='r',dash_capstyle='round',solid_capstyle='round',label='DFT')     
        pylab.errorbar(means[2],means[3],xerr=emeans[2],yerr=emeans[3],capsize=20,color='g',dash_capstyle='round',solid_capstyle='round',label='Taylor')  
        pylab.errorbar(means[4],means[5],xerr=emeans[4],yerr=emeans[5],capsize=20,color='b',dash_capstyle='round',solid_capstyle='round',label='Gaussian')
        pylab.errorbar(means[6],means[7],xerr=emeans[6],yerr=emeans[7],capsize=20,color='m',dash_capstyle='round',solid_capstyle='round',label='$\\chi^2$')
        pylab.errorbar(means[0],means[1],xerr=stds[0],yerr=stds[1],capsize=10,color='r',linestyle='--',linewidth=5)
        pylab.errorbar(means[2],means[3],xerr=stds[2],yerr=stds[3],capsize=10,color='g',linestyle='--',linewidth=5)
        pylab.errorbar(means[4],means[5],xerr=stds[4],yerr=stds[5],capsize=10,color='b',linestyle='--',linewidth=5)
        pylab.errorbar(means[6],means[7],xerr=stds[6],yerr=stds[7],capsize=10,color='m',linestyle='--',linewidth=5)
        if legend:
            pylab.legend(loc='best')



except ImportError:
    pass

doplots=False
if doplots:
    print "Showing some nice plots"

    from pylab import *
    figure(1)
    clf()
    accuracy_plot(amp=10000)
    accuracy_plot(amp=1000)
    accuracy_plot(amp=100)
    accuracy_plot(amp=20)
    accuracy_plot(width=2., amp=10000)
    accuracy_plot(width=2., amp=1000)
    accuracy_plot(width=2., amp=100)
    legend(loc='best')
    xlabel("Upsample Factor")
    ylabel("Real offset - measured offset")

    figure(2)
    clf()
    title("Extended Structure")
    accuracy_plot_extended(noise=10**-2)
    accuracy_plot_extended(noise=10**-1)
    accuracy_plot_extended(noise=10**-0)
    legend(loc='best')
    xlabel("Upsample Factor")
    ylabel("Real offset - measured offset")


    # some neat test codes:
    # compare_offsets = compare_methods(testim, testim_offset, nthreads=8)
    # plot(compare_offsets[:,0],compare_offsets[:,2],'.')
    # errorbar(compare_offsets[:,0].mean(),compare_offsets[:,2].mean(),xerr=compare_offsets[:,0].std(),yerr=compare_offsets[:,2].std(),marker='x',linestyle='none')
    # plot(compare_offsets[:,1],compare_offsets[:,3],'.')
    # errorbar(compare_offsets[:,1].mean(),compare_offsets[:,3].mean(),xerr=compare_offsets[:,1].std(),yerr=compare_offsets[:,3].std(),marker='x',linestyle='none')
