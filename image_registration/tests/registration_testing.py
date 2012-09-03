try:
    from AG_image_tools.cross_correlation_shifts import cross_correlation_shifts
    from AG_image_tools.register_images import register_images
    from AG_image_tools.chi2_shifts import chi2_shift
    from AG_fft_tools import dftups,upsample_image,shift
except ImportError:
    from image_registration.cross_correlation_shifts import cross_correlation_shifts
    from image_registration.register_images import register_images
    from image_registration.chi2_shifts import chi2_shift
    from image_registration.fft_tools import dftups,upsample_image,shift

import numpy as np

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
            xoff,yoff,reg = register_images(
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
            dy,dx = do_n_extended_fits(1,xsh,ysh, imsize, shift_func=register_images,sfkwargs={'usfac':usf},noise=noise)[0]
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
            shift_func=register_images, sfkwargs={'usfac':usf,'maxoff':maxoff}, noise=noise))

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
            dy,dx = register_images(im1,im2,usfac=usf,**kwargs)
            offsets.append([dx,dy])

        return np.array(offsets)

    def register_noise_test(im1,im2, ntests=100, noise=np.std,
            register_method=register_images, return_error=False, **kwargs):
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
            if return_error:
                dx,dy,edx,edy = register_method(im1,im2+extra_noise,return_error=True,**kwargs)
                offsets.append([dx,dy,edx,edy])
            else:
                dx,dy = register_method(im1,im2+extra_noise,return_error=False,**kwargs)
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
            dxr, dyr, edxr, edyr = register_images(im1, im2+extra_noise, usfac=usfac,
                    return_error=True, **kwargs)
            dxccs, dyccs, edxccs, edyccs = cross_correlation_shifts(im1,
                    im2+extra_noise,
                    errim2=im2*0+extra_noise.std(),
                    return_error=True, **kwargs)
            dxccg, dyccg, edxccg, edyccg = cross_correlation_shifts(im1,
                    im2+extra_noise, return_error=True, gaussfit=True,
                    **kwargs)
            dxchi, dychi, edxchi, edychi = chi2_shift(im1, im2+extra_noise,
                    err=im2*0+noise,
                    return_error=True, upsample_factor='auto', verbose=True, **kwargs)
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

        print "Standard Deviations: ", stds
        print "Error Means: ", emeans
        print "emeans/stds: ", emeans/stds

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
