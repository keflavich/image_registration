import numpy as np
try:
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
except ImportError:
    import pyfits
    import pywcs

def fits_overlap(file1,file2):
    """
    Create a header containing the exact overlap region between two .fits files

    Does NOT check to make sure the FITS files are in the same coordinate system!
    """

    hdr1 = pyfits.getheader(file1)
    hdr2 = pyfits.getheader(file2)

    wcs1 = pywcs.WCS(hdr1)
    wcs2 = pywcs.WCS(hdr2)

    ((xmax1,ymax1),) = wcs1.wcs_pix2world([[hdr1['NAXIS1'],hdr1['NAXIS2']]],1)
    ((xmax2,ymax2),) = wcs2.wcs_pix2world([[hdr2['NAXIS1'],hdr2['NAXIS2']]],1)

    ((xmin1,ymin1),) = wcs1.wcs_pix2world([[1,1]],1)
    ((xmin2,ymin2),) = wcs2.wcs_pix2world([[1,1]],1)

    xmin = min(xmin1,xmin2)
    xmax = max(xmax1,xmax2)
    ymin = min(ymin1,ymin2)
    ymax = max(ymax1,ymax2)

    try:
        cdelt1,cdelt2 = np.abs(np.vstack([wcs1.wcs.cd.diagonal(), wcs2.wcs.cd.diagonal()])).min(axis=0) * np.sign(wcs1.wcs.cd).diagonal()
    except AttributeError:
        cdelt1,cdelt2 = np.abs(np.vstack([wcs1.wcs.cdelt, wcs2.wcs.cdelt])).min(axis=0) * np.sign(wcs1.wcs.cdelt)

    # may want to change this later...
    new_header = hdr1
    new_header['CRVAL1'] = (xmin+xmax)/2.
    new_header['CRVAL2'] = (ymin+ymax)/2.
    new_header['CDELT1'] = cdelt1
    new_header['CDELT2'] = cdelt2
    new_header['NAXIS1'] = np.ceil(np.abs((xmax-xmin)/cdelt1))
    new_header['NAXIS2'] = np.ceil(np.abs((ymax-ymin)/cdelt2))
    new_header['CRPIX1'] = new_header['NAXIS1']/2
    new_header['CRPIX2'] = new_header['NAXIS2']/2

    return new_header
