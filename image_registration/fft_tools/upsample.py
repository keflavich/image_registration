import fast_ffts
import warnings
import numpy as np
import scale
import shift
import zoom

def dftups(inp,nor=None,noc=None,usfac=1,roff=0,coff=0):
    """
    *translated from matlab*
    http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

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
    from numpy.fft import ifftshift,fftfreq
    from numpy import pi,newaxis,floor

    nr,nc=np.shape(inp);
    # Set defaults
    if noc is None: noc=nc;
    if nor is None: nor=nr;
    # Compute kernels and obtain DFT by matrix products
    term1c = ( ifftshift(np.arange(nc,dtype='float') - floor(nc/2)).T[:,newaxis] )/nc # fftfreq
    term2c = (( np.arange(noc,dtype='float') - coff  )/usfac)[newaxis,:]              # output points
    kernc=np.exp((-1j*2*pi)*term1c*term2c);

    term1r = ( np.arange(nor,dtype='float').T - roff )[:,newaxis]                # output points
    term2r = ( ifftshift(np.arange(nr,dtype='float')) - floor(nr/2) )[newaxis,:] # fftfreq
    kernr=np.exp((-1j*2*pi/(nr*usfac))*term1r*term2r);
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    out=np.dot(np.dot(kernr,inp),kernc);
    #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
    return out 

def dftups1d(inp,usfac=1,outsize=None,offset=0, return_xouts=False):
    """
    """
    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab
    from numpy.fft import ifftshift,fftfreq
    from numpy import pi,newaxis,floor

    insize, = inp.shape
    if outsize is None: outsize=insize

    # Compute kernel and obtain DFT by matrix products
    term1 = fftfreq(insize)[:,newaxis]
    term2 = ((np.arange(outsize,dtype='float') + offset)/usfac)[newaxis,:]
    kern=np.exp((-1j*2*pi)*term1*term2);
    # Without the +1 in term 2, the output is always shifted by 1.
    # But, the actual X-axis starts at zero, so I have to subtract 1 below
    # This is weird... it implies that the FT is indexed from 1, which is not
    # a meaningful statement
    # My best guess is that it's a problem with e^0, the mean case, throwing things
    # off, but that's not a useful statement

    out = np.dot(inp,kern)
    if return_xouts:
        return term2.squeeze(),out
    return out 


def upsample_image(image, upsample_factor=1, output_size=None, nthreads=1,
        use_numpy_fft=False, xshift=0, yshift=0):
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

def odddftups(inp,nor=None,noc=None,usfac=1,roff=0,coff=0):
    from numpy.fft import ifftshift
    from numpy import pi,newaxis,floor

    nr,nc=np.shape(inp);

    # Set defaults
    if noc is None: noc=nc;
    if nor is None: nor=nr;

    if nr % 2 == 1:
        oddr = True
        nrnew = nr+1
    else:
        oddr = False
    if nr % 2 == 1:
        oddr = True
        nrnew = nr+1
    else:
        oddr = False

    # Compute kernels and obtain DFT by matrix products
    term1c = ( ifftshift(np.arange(nc) - floor(nc/2)).T[:,newaxis] )
    term2c = ( np.arange(noc) - coff  )[newaxis,:]
    kernc=np.exp((-1j*2*pi/(nc*usfac))*term1c*term2c);
    term1r = ( np.arange(nor).T - roff )[:,newaxis]
    term2r = ( ifftshift(np.arange(nr)) - floor(nr/2) )[newaxis,:]
    kernr=np.exp((-1j*2*pi/(nr*usfac))*term1r*term2r);
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    out=np.dot(np.dot(kernr,inp),kernc);
    #return np.roll(np.roll(out,+1,axis=0),+1,axis=1)
    return out 


def dftups1d(inp,nor=None,usfac=1,roff=0):
    """
    1D upsampling... not exactly dft becuase I still don't understand it =(
    """
    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab
    from numpy.fft import ifftshift
    #from numpy import pi,newaxis,floor
    from scipy.signal import resample


    nr=np.size(inp);
    newsize = nr * usfac
    #shifted = shift(inp, roff, mode='wrap')
    shifted = shift.shift1d(inp,roff)
    ups = resample(shifted.astype('float'),newsize)
    lolim = nr/2-nr/2
    uplim = nr/2+nr/2
    # I think it would always have to be wrong on the upper side
    if uplim-lolim > nr:
        uplim -= 1
    elif uplim-lolim < nr:
        uplim += 1
    if uplim - lolim != nr: raise ValueError('impossible?')
    out = ups[lolim:uplim]

    #oldx = np.arange(nr)
    #newx = np.linspace(nr/2.-nr/2./usfac+roff/usfac,nr/2.+nr/2./usfac+roff/usfac,nr)
    #oldx = np.linspace(0,1,nr)
    #newx = np.linspace(0,1,newsize)
    #inshift = shift.shift1d(inp,roff)
    #out = ups = np.interp(newx,oldx,np.real(inp))

    #lolim = newsize/2+roff*usfac-nr/2
    #uplim = newsize/2+roff*usfac+nr/2
    #out = ups[lolim:uplim]
    
    # Set defaults
    #if nor is None: nor=nr;
    # Compute kernels and obtain DFT by matrix products
    #kernc=np.exp((-1j*2*pi/(nc*usfac))*( ifftshift(np.arange(nc) - floor(nc/2)).T[:,newaxis] )*( np.arange(noc) - coff  )[newaxis,:]);
    #kernr=np.exp((-1j*2*pi/(nr*usfac))*( np.arange(nor).T - roff )[:,newaxis]*( ifftshift(np.arange(nr)) - floor(nr/2) )[newaxis,:]);
    #kernc=np.ones(nr,dtype='float')/float(nr)
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    #out=np.dot(kernr,inp)
    #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
    return out 

if __name__ == "__main__" and False:

    # breakdown of the dft upsampling method
    from numpy.fft import ifftshift
    from numpy import pi,newaxis,floor

    from pylab import *
    
    xx,yy = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
    # a Gaussian image
    data = np.exp(-(xx**2+yy**2)/(0.5**2 * 2.))
    fftn,ifftn = fast_ffts.get_ffts(nthreads=4, use_numpy_fft=False)
    print "input max pixel: ",np.unravel_index(data.argmax(),data.shape)
    inp = ifftn(data)

    nr,nc=np.shape(inp);
    noc,nor = nc,nr # these are the output sizes

    # upsample_factor
    usfac = 20.
    for usfac in [1,2,5,10,20,30,40]:
        # the "virtual image" will have size im.shape[0]*usfac,im.shape[1]*usfac
        # To "zoom in" on the center of the image, we need an offset that identifies
        # the lower-left corner of the new image
        vshape = inp.shape[0]*usfac,inp.shape[1]*usfac
        roff = -(vshape[0] - usfac - nor)/2. -1
        coff = -(vshape[1] - usfac - noc)/2. -1

        # shifts decided automatically now
        # roff,coff = 0,0 # -50,-50


        # Compute kernels and obtain DFT by matrix products
        term1c = ( ifftshift(np.arange(nc) - floor(nc/2)).T[:,newaxis] )
        term2c = ( np.arange(noc) - coff  )[newaxis,:]
        kernc=np.exp((-1j*2*pi/(nc*usfac))*term1c*term2c);
        
        figure(1)
        clf()
        subplot(121)
        imshow(term1c)
        title("term1 (col)")
        colorbar()
        subplot(122)
        imshow(term2c)
        title("term2 (col)")
        colorbar()


        term1r = ( np.arange(nor).T - roff )[:,newaxis]
        term2r = ( ifftshift(np.arange(nr)) - floor(nr/2) )[newaxis,:]
        kernr=np.exp((-1j*2*pi/(nr*usfac))*term1r*term2r);

        figure(2)
        clf()
        subplot(121)
        imshow(term1r)
        title("term1 (row)")
        colorbar()
        subplot(122)
        imshow(term2r)
        title("term2 (row)")
        colorbar()

        figure(3)
        clf()
        subplot(131)
        imshow(np.abs(kernr))
        subplot(132)
        imshow(kernr.real)
        subplot(133)
        imshow(kernr.imag)

        #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
        #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
        dot1 = np.dot(kernr,inp)
        out=np.dot(dot1,kernc);

        # http://stackoverflow.com/a/9479621/814354
        # wrong from scipy.linalg import fblas as FB
        # wrong out2 = FB.dgemm(alpha=1.0, a=dot1, b=kernc)

        figure(10)
        subplot(121)
        imshow(data)
        title("gaussian")
        subplot(122)
        imshow(np.abs(out))
        title('zoomed')

        print "usfac: ",usfac,"max pixel: ",np.unravel_index(np.abs(out).argmax(),out.shape)

        figure(11)
        clf()
        imshow(np.abs(dftups(inp,inp.shape[0]*2,inp.shape[1]*2,usfac=2)))

