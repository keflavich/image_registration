import fast_ffts
import warnings
import numpy as np

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
        *ADAM'S NOTE: ifftshift appeared to be incorrect; fftshift is right
            (only different for odd-shaped images).  Except, doesn't affect
            the part I thought it did, so ifft it will be!
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
    term1c = ( ifftshift(np.arange(nc) - floor(nc/2)).T[:,newaxis] )
    term2c = ( np.arange(noc) - coff  )[newaxis,:]
    kernc=np.exp((-1j*2*pi/(nc*usfac))*term1c*term2c);
    term1r = ( np.arange(nor).T - roff )[:,newaxis]
    term2r = ( ifftshift(np.arange(nr)) - floor(nr/2) )[newaxis,:]
    kernr=np.exp((-1j*2*pi/(nr*usfac))*term1r*term2r);
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    out=np.dot(np.dot(kernr,inp),kernc);
    #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
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

def center_zoom_image(image, upsample_factor=1, output_size=None, nthreads=1,
        use_numpy_fft=False, xshift=0, yshift=0, return_axes=False):
    """
    Same as :func:`upsample_image` but with "default" xoff/yoff computed such that the zoom
    always remains in the center
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

    #vshape = image.shape[0]*upsample_factor,image.shape[1]*upsample_factor
    # int(float(a)/b) is the "round towards zero" division operation
    #roff = -((vshape[0] - upsample_factor - s1)/2) -1#*(s1 % 2)
    #coff = -((vshape[1] - upsample_factor - s2)/2) -1#*(s2 % 2)
    # worked for the odd case
    #roff = -int(np.round(float(vshape[0] - upsample_factor - s1)/2.)) - (upsample_factor%2==0) 
    #coff = -int(np.round(float(vshape[1] - upsample_factor - s2)/2.)) - (upsample_factor%2==0) 
    # in principle, works for even case
    # this is the solution I found on paper.
    # offset_from_bottom_left_corner_Y = round((image.shape[0]*upsample_factor  - s1)/2.) 
    # offset_from_bottom_left_corner_X = round((image.shape[1]*upsample_factor  - s2)/2.) 
    # this is STILL WRONG roff = -int(np.round(float(image.shape[0]*upsample_factor - s1)/2.))
    # this is STILL WRONG coff = -int(np.round(float(image.shape[1]*upsample_factor - s2)/2.))

    # round((image.shape[0]*upsample_factor  - s1)/2.) is the size of the big
    # image (with split pixels) minus the size of the zoomed-in image (in split
    # pixel units) divided by two because there is a left and a right buffer
    # The added factor upsample_factor/2*(image.shape[0]%2==1) is to deal with odd-shaped images,
    # which for no particularly obvious reason are mistreated by dftups...
    # The last term is if the input and output image shapes differ in even/oddness, the zoom should be offset by half a pixel...
    roff = -round((image.shape[0]*upsample_factor  - s1)/2.) + upsample_factor/2*(image.shape[0]%2==1) 
    coff = -round((image.shape[1]*upsample_factor  - s2)/2.) + upsample_factor/2*(image.shape[1]%2==1) 

    # doesn't this look like a hack?  It feels like a hack.
    roff += 0.5 * (image.shape[0]%2==0) * (upsample_factor%2==0) + (image.shape[0]%2==0)*((upsample_factor-1)/2)
    coff += 0.5 * (image.shape[1]%2==0) * (upsample_factor%2==0) + (image.shape[1]%2==0)*((upsample_factor-1)/2)
    #roff += -((image.shape[0]-s1)%2==1)*(upsample_factor%2==0)*(image.shape[0]<s1) #*((image.shape[0]-s1)%2==1)*(image.shape[0]<s1)
    #coff += -((image.shape[1]-s2)%2==1)*(upsample_factor%2==0)*(image.shape[1]<s2) #*((image.shape[1]-s2)%2==1)*(image.shape[1]<s2)
    print "roff,coff,upsample_factor,shape: ",roff,coff,upsample_factor,image.shape

    # discovered mostly by guess and check (for shame):
    # yshift/xshift must be scale up by upsample factor because
    # they get scaled with the image
    ups = dftups(imfft, s1, s2, upsample_factor, 
            roff=roff-yshift*upsample_factor, 
            coff=coff-xshift*upsample_factor)

    if return_axes:
        yy,xx = np.indices([s1,s2],dtype='float')
        xshifts_corrections = (xx-coff-xshift)/upsample_factor + xshift #+ (vshape[1]%2==0) * 1./(2*upsample_factor)
        yshifts_corrections = (yy-roff-yshift)/upsample_factor + yshift #+ (vshape[0]%2==0) * 1./(2*upsample_factor)
        # black = (red - (ups-1)/2)/ups
        xshifts_corrections = (xx-coff)/upsample_factor - 0.5 + 1/(2.*upsample_factor)-xshift
        yshifts_corrections = (yy-roff)/upsample_factor - 0.5 + 1/(2.*upsample_factor)-yshift
        #yyOrig,xxOrig = np.linspace(0,image.shape[0]-1,s1),np.linspace(0,image.shape[1]-1,s2)
        #yy,xx = np.meshgrid(yyOrig,xxOrig)
        #xshifts_corrections = (xx*upsample_factor + (upsample_factor-1)/2. - coff - xshift*upsample_factor)
        #yshifts_corrections = (yy*upsample_factor + (upsample_factor-1)/2. - roff - yshift*upsample_factor)
        return xshifts_corrections,yshifts_corrections,np.abs(ups)

    return np.abs(ups)


if __name__ == "__main__":

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

