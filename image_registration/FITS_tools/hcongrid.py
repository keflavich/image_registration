import numpy as np
try:
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
except ImportError:
    import pyfits
    import pywcs
try:
    import scipy.ndimage

    def hcongrid(image, header1, header2, preserve_bad_pixels=True, **kwargs):
        """
        Interpolate an image from one FITS header onto another

        kwargs will be passed to `scipy.ndimage.map_coordinates`

        Parameters
        ----------
        image : ndarray
            A two-dimensional image 
        header1 : `pyfits.Header` or `pywcs.WCS`
            The header or WCS corresponding to the image
        header2 : `pyfits.Header` or `pywcs.WCS`
            The header or WCS to interpolate onto
        preserve_bad_pixels: bool
            Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
            pixels will be set to zero

        Returns
        -------
        ndarray with shape defined by header2's naxis1/naxis2

        Raises
        ------
        TypeError if either is not a Header or WCS instance
        Exception if image1's shape doesn't match header1's naxis1/naxis2

        Examples
        --------
        (not written with >>> because test.fits/test2.fits do not exist)
        fits1 = pyfits.open('test.fits')
        target_header = pyfits.getheader('test2.fits')
        new_image = hcongrid(fits1[0].data, fits1[0].header, target_header)

        """

        if issubclass(pywcs.WCS, header1.__class__):
            wcs1 = header1
        else:
            try:
                wcs1 = pywcs.WCS(header1)
            except:
                raise TypeError("Header1 must either be a pyfits.Header or pywcs.WCS instance")

        if not (wcs1.naxis1 == image.shape[1] and wcs1.naxis2 == image.shape[0]):
            raise Exception("Image shape must match header shape.")

        if issubclass(pywcs.WCS, header2.__class__):
            wcs2 = header2
        else:
            try:
                wcs2 = pywcs.WCS(header2)
            except:
                raise TypeError("Header2 must either be a pyfits.Header or pywcs.WCS instance")

        if not all([w1==w2 for w1,w2 in zip(wcs1.wcs.ctype,wcs2.wcs.ctype)]):
            # do unit conversions
            raise NotImplementedError("Unit conversions have not yet been implemented.")

        # sigh... why does numpy use matrix convention?  Makes everything so much harder...
        outshape = [wcs2.naxis2,wcs2.naxis1]
        yy2,xx2 = np.indices(outshape)
        lon2,lat2 = wcs2.wcs_pix2sky(xx2, yy2, 0)
        xx1,yy1 = wcs1.wcs_sky2pix(lon2, lat2, 0)
        grid1 = np.array([yy1.reshape(outshape),xx1.reshape(outshape)])

        bad_pixels = np.isnan(image) + np.isinf(image)

        image[bad_pixels] = 0

        newimage = scipy.ndimage.map_coordinates(image, grid1, **kwargs)

        if preserve_bad_pixels:
            newbad = scipy.ndimage.map_coordinates(bad_pixels, grid1, order=0, mode='constant', cval=np.nan)
            newimage[newbad] = np.nan
        
        return newimage

    def zoom_fits(fitsfile, scalefactor, preserve_bad_pixels=True, **kwargs):
        """
        Zoom in on a FITS image by interpolating using scipy.ndimage.zoom

        Parameters
        ----------
        fitsfile: str
            FITS file name
        scalefactor: float
            Zoom factor along all axes
        preserve_bad_pixels: bool
            Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
            pixels will be set to zero
        """

        arr = pyfits.getdata(fitsfile)
        h = pyfits.getheader(fitsfile)

        h['CRPIX1'] = (h['CRPIX1']-1)*scalefactor + scalefactor/2. + 0.5
        h['CRPIX2'] = (h['CRPIX2']-1)*scalefactor + scalefactor/2. + 0.5
        if 'CD1_1' in h:
            for ii in (1,2):
                for jj in (1,2):
                    k = "CD%i_%i" % (ii,jj)
                    if k in h: # allow for CD1_1 but not CD1_2
                        h[k] = h[k]/scalefactor
        elif 'CDELT1' in h:
            h['CDELT1'] = h['CDELT1']/scalefactor
            h['CDELT2'] = h['CDELT2']/scalefactor

        bad_pixels = np.isnan(arr) + np.isinf(arr)

        arr[bad_pixels] = 0

        upscaled = scipy.ndimage.zoom(arr,scalefactor,**kwargs)

        if preserve_bad_pixels:
            bp_up = scipy.ndimage.zoom(bad_pixels,scalefactor,mode='constant',cval=np.nan,order=0)
            upscaled[bp_up] = np.nan

        up_hdu = pyfits.PrimaryHDU(data=upscaled, header=h)
        
        return up_hdu

except ImportError:
    # needed to do this to get travis-ci tests to pass, even though scipy is installed...
    def hcongrid(*args, **kwargs):
        raise ImportError("scipy.ndimage could not be imported; hcongrid is not available")

    def zoom_fits(*args, **kwargs):
        raise ImportError("scipy.ndimage could not be imported; zoom_fits is not available")
