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

    Parameters
    ----------
    file1,file2 : str,str
        files from which to extract header strings
    """

    hdr1 = pyfits.getheader(file1)
    hdr2 = pyfits.getheader(file2)
    return header_overlap(hdr1,hdr2)

def header_overlap(hdr1,hdr2,max_separation=180,overlap="union"):
    """
    Create a header containing the exact overlap region between two .fits files

    Does NOT check to make sure the FITS files are in the same coordinate system!

    Parameters
    ----------
    hdr1,hdr2 : pyfits.Header
        Two pyfits headers to compare
    max_separation : int
        Maximum number of degrees between two headers to consider before flipping
        signs on one of them (this to deal with the longitude=0 region)
    overlap: 'union' or 'intersection'
        Which merger to do
    """
    wcs1 = pywcs.WCS(hdr1)
    wcs2 = pywcs.WCS(hdr2)

    ((xmax1,ymax1),) = wcs1.wcs_pix2sky([[hdr1['NAXIS1'],hdr1['NAXIS2']]],1)
    ((xmax2,ymax2),) = wcs2.wcs_pix2sky([[hdr2['NAXIS1'],hdr2['NAXIS2']]],1)

    ((xmin1,ymin1),) = wcs1.wcs_pix2sky([[1,1]],1)
    ((xmin2,ymin2),) = wcs2.wcs_pix2sky([[1,1]],1)

    # make sure the edges are all in the same quadrant-ish
    xmlist = [ xm - 360 if xm > max_separation else 
            xm + 360 if xm < -max_separation else xm
        for xm in xmin1,xmax1,xmin2,xmax2]
    xmin1,xmax1,xmin2,xmax2 = xmlist

    # check signs
    if xmin2 > xmax2:
        xmax2,xmin2 = xmin2,xmax2
    if xmin1 > xmax1:
        xmax1,xmin1 = xmin1,xmax1
    if ymin2 > ymax2:
        ymax2,ymin2 = ymin2,ymax2
    if ymin1 > ymax1:
        ymax1,ymin1 = ymin1,ymax1

    if overlap=='union':
        xmin = min(xmin1,xmin2)
        xmax = max(xmax1,xmax2)
        ymin = min(ymin1,ymin2)
        ymax = max(ymax1,ymax2)
    elif overlap=='intersection':
        xmin = max(xmin1,xmin2)
        xmax = min(xmax1,xmax2)
        ymin = max(ymin1,ymin2)
        ymax = min(ymax1,ymax2)
    else:
        raise ValueError("Overlap must be 'union' or 'intersection'")

    try:
        cdelt1,cdelt2 = np.abs(np.vstack([wcs1.wcs.cd.diagonal(), wcs2.wcs.cd.diagonal()])).min(axis=0) * np.sign(wcs1.wcs.cd).diagonal()
    except AttributeError:
        cdelt1,cdelt2 = np.abs(np.vstack([wcs1.wcs.cdelt, wcs2.wcs.cdelt])).min(axis=0) * np.sign(wcs1.wcs.cdelt)

    # no overlap at all
    if ((xmin1 > xmax2) or 
        (xmin2 > xmax1)):
        naxis1 = 0
    else:
        naxis1 = np.ceil(np.abs((xmax-xmin)/cdelt1))
    if ymin1 > ymax2 or ymin2 > ymax1:
        naxis2 = 0
    else:
        naxis2 = np.ceil(np.abs((ymax-ymin)/cdelt2))


    # may want to change this later...
    new_header = hdr1.copy()
    new_header['CRVAL1'] = (xmin+xmax)/2.
    new_header['CRVAL2'] = (ymin+ymax)/2.
    new_header['CDELT1'] = cdelt1
    new_header['CDELT2'] = cdelt2
    new_header['NAXIS1'] = int(naxis1)
    new_header['NAXIS2'] = int(naxis2)
    new_header['CRPIX1'] = new_header['NAXIS1']/2.
    new_header['CRPIX2'] = new_header['NAXIS2']/2.

    return new_header
