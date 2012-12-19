import numpy
try:
    from scipy.stats import nanmean as mean
except ImportError:
    from numpy import mean

def downsample(myarr,factor,estimator=mean):
    """
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.

    This code is pure numpy and should be fast.

    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( numpy.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr

def downsample_cube(myarr,factor,ignoredim=0):
    """
    Downsample a 3D array by averaging over *factor* pixels on the last two
    axes.
    """
    if ignoredim > 0: myarr = myarr.swapaxes(0,ignoredim)
    zs,ys,xs = myarr.shape
    crarr = myarr[:,:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = mean(numpy.concatenate([[crarr[:,i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    if ignoredim > 0: dsarr = dsarr.swapaxes(0,ignoredim)
    return dsarr

def downsample_1d(myarr,factor,estimator=mean):
    """
    Downsample a 1D array by averaging over *factor* pixels.
    Crops right side if the shape is not a multiple of factor.

    This code is pure numpy and should be fast.

    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    xs = myarr.shape
    crarr = myarr[:xs-(xs % int(factor))]
    dsarr = estimator( numpy.concatenate([[crarr[i::factor] 
        for i in range(factor)] ]),axis=0)
    return dsarr

