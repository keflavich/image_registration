from ..fft_tools import zoom
import numpy as np
import pytest
import itertools

def gaussian(x):
    return np.exp(-x**2/2.)

def measure_difference_zoom_samesize(imsize, upsample_factor,doplot=False,ndim=2):
    """
    Test that zooming in by some factor with the same input & output sizes
    works
    """
    inds = np.indices([imsize]*ndim)
    rr = ((inds-(imsize-1)/2.)**2).sum(axis=0)**0.5
    gg = gaussian(rr)
    xz,zz = zoom.zoomnd(gg,upsample_factor,return_xouts=True)
    xr = ((xz - (imsize-1.)/2.)**2).sum(axis=0)**0.5

    return ((gaussian(xr)-zz)**2).sum() 

def measure_zoom_fullsize(imsize, upsample_factor,doplot=False,ndim=2):
    """
    """
    inds = np.indices([imsize]*ndim)
    rr = ((inds-(imsize-1)/2.)**2).sum(axis=0)**0.5
    gg = gaussian(rr)
    outshape = [s*upsample_factor for s in gg.shape]
    xz,zz = zoom.zoomnd(gg,upsample_factor,outshape=outshape,return_xouts=True)
    xr = ((xz - (imsize-1.)/2.)**2).sum(axis=0)**0.5
                          
    return ((gaussian(xr)-zz)**2).sum() 



def measurements(imsizes,upsample_factors,accuracies):
    import pylab as pl
    pl.figure(1)
    pl.clf()
    pl.pcolormesh(imsizes,upsample_factors,accuracies)
    pl.colorbar()
    pl.xlabel("Upsample Factor")
    pl.ylabel("Image Size")

    pl.figure(2)
    pl.clf()
    pl.plot(upsample_factors,accuracies[:,1::2])
    pl.xlabel("Upsample Factor (even imsize)")
    pl.ylabel("Accuracy")

    pl.figure(3)
    pl.clf()
    pl.plot(upsample_factors,accuracies[:,::2])
    pl.xlabel("Upsample Factor (odd imsize)")
    pl.ylabel("Accuracy")

    pl.figure(4)
    pl.clf()
    pl.plot(imsizes[::2],accuracies.T[::2,:])
    pl.xlabel("Image Sizes (odd)")
    pl.ylabel("Accuracy")

    pl.figure(5)
    pl.clf()
    pl.plot(imsizes[1::2],accuracies.T[1::2,:])
    pl.xlabel("Image Sizes (even)")
    pl.ylabel("Accuracy")

    def model_accuracy(x, power, const):
        return const*x**power

    import scipy.optimize as scopt


    pl.figure(6)
    pl.clf()
    pl.plot(upsample_factors,accuracies[:,::2].max(axis=1),label='Odd Imsize')
    pl.plot(upsample_factors,accuracies[:,1::2].max(axis=1),label='Even Imsize')
    pl.xlabel("Upsample Factor")
    pl.ylabel("Worst Accuracy")
    pl.legend(loc='best')
    oddpars,err=scopt.curve_fit(model_accuracy,upsample_factors,accuracies[:,::2].max(axis=1),maxfev=2000,p0=[1.5,1/2000.])
    evenpars,err=scopt.curve_fit(model_accuracy,upsample_factors,accuracies[:,1::2].max(axis=1),maxfev=2000)
    pl.plot(upsample_factors,model_accuracy(upsample_factors,*oddpars),label='Odd fit',linestyle='--',color='k')
    pl.plot(upsample_factors,model_accuracy(upsample_factors,*evenpars),label='Even fit',linestyle=':',color='k')
    pl.plot(upsample_factors,0.002+model_accuracy(upsample_factors,*oddpars),label='Odd fit',linestyle='--',color='r',linewidth=2)
    pl.plot(upsample_factors,0.002+model_accuracy(upsample_factors,*evenpars),label='Even fit',linestyle=':',color='r',linewidth=2)
    print "odd (upsample): ",oddpars
    print "even (upsample): ",evenpars


    pl.figure(7)
    pl.clf()
    pl.plot(imsizes[::2],accuracies[:,::2].max(axis=0),label='Odd Imsize')
    pl.plot(imsizes[1::2],accuracies[:,1::2].max(axis=0),label='Even Imsize')
    pl.plot(imsizes[19::24],accuracies[:,19::24].max(axis=0),label='Worst Imsize',linestyle='none',marker='s')
    pl.xlabel("Image Size")
    pl.ylabel("Worst Accuracy")
    pl.legend(loc='best')
    oddpars,err=scopt.curve_fit(model_accuracy,imsizes[::2],accuracies[:,::2].max(axis=0),maxfev=2000)
    evenpars,err=scopt.curve_fit(model_accuracy,imsizes[1::2],accuracies[:,1::2].max(axis=0),maxfev=2000)
    pl.plot(imsizes[::2],model_accuracy(imsizes[::2],*oddpars),label='Odd fit',linestyle='--',color='k')
    pl.plot(imsizes[1::2],model_accuracy(imsizes[1::2],*evenpars),label='Even fit',linestyle=':',color='k')
    print "odd (imsize): ",oddpars
    print "even (imsize): ",evenpars
    worstevenpars,err=scopt.curve_fit(model_accuracy,imsizes[19::24],accuracies[:,19::24].max(axis=0),maxfev=2000)
    print "worst evenpars: ",worstevenpars
    pl.plot(imsizes,model_accuracy(imsizes,*worstevenpars),label='Worst Even fit',linestyle='-',color='r')
    pl.plot(imsizes,0.01+model_accuracy(imsizes,*worstevenpars),label='Worst Even fit',linestyle='--',color='r')

if __name__ == "__main__":

    imsizes = np.arange(5,201)
    upsample_factors = np.arange(1,25)
    
    accuracies = np.array([[measure_difference_zoom_samesize(sz, us)
            for sz in imsizes] for us in upsample_factors])

    measurements(imsizes,upsample_factors,accuracies)

    imsizes = np.arange(5,49)
    upsample_factors = np.arange(1,20)
    accuracies_fullzoom = np.array([[measure_zoom_fullsize(sz, us)
            for sz in imsizes] for us in upsample_factors])

    measurements(imsizes,upsample_factors,accuracies_fullzoom)

