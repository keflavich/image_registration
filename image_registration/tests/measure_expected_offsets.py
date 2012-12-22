from image_registration.cross_correlation_shifts import cross_correlation_shifts
from image_registration.register_images import register_images
from image_registration import chi2_shifts
from image_registration.fft_tools import dftups,upsample_image,shift,smooth
from image_registration.tests import registration_testing as rt

import numpy as np
import matplotlib.pyplot as pl
import time
from functools import wraps

def print_timing(func):
    @wraps(func)
    def wrapper(*arg,**kwargs):
        t1 = time.time()
        res = func(*arg,**kwargs)
        t2 = time.time()
        print '%s took %0.5g s' % (func.func_name, (t2-t1))
        return res
    return wrapper



def test_measure_offsets(xsh, ysh, imsize, noise_taper=False, noise=0.5, chi2_shift=chi2_shifts.chi2_shift):
    image = rt.make_extended(imsize)
    offset_image = rt.make_offset_extended(image, xsh, ysh, noise_taper=noise_taper, noise=noise)
    if noise_taper:
        noise = noise/rt.edge_weight(imsize)
    else:
        noise = noise

    return chi2_shift(image,offset_image,noise,return_error=True,upsample_factor='auto')

@print_timing
def montecarlo_test_offsets(xsh, ysh, imsize, noise_taper=False, noise=0.5, nsamples=100, **kwargs):

    results = [test_measure_offsets(xsh, ysh, imsize, noise_taper=noise_taper,
                                    noise=noise, **kwargs)
                for ii in xrange(nsamples)]

    xoff,yoff,exoff,eyoff = zip(*results)

    return xoff,yoff,exoff,eyoff

def plot_montecarlo_test_offsets(xsh, ysh, imsize, noise=0.5, name="", **kwargs):
    xoff,yoff,exoff,eyoff = montecarlo_test_offsets(xsh,ysh,imsize,noise=noise,**kwargs)
    pl.plot(xsh,ysh,'k+')
    means = [np.mean(x) for x in (xoff,yoff,exoff,eyoff)]
    stds = [np.std(x) for x in (xoff,yoff,exoff,eyoff)]
    pl.plot(xoff,yoff,',',label=name)
    pl.errorbar(means[0],means[1],xerr=stds[0],yerr=stds[1],label=name+"$\mu+\sigma$")
    pl.errorbar(means[0],means[1],xerr=means[2],yerr=means[3],label=name+"$\mu+$\mu(\sigma)$")
    #pl.legend(loc='best')

    return xoff,yoff,exoff,eyoff,means,stds

def montecarlo_tests_of_imsize(xsh,ysh,imsizerange=[15,105,5],noise=0.5, figstart=0, **kwargs):
    """
    Perform many monte-carlo tests as a function of the image size
    """
    pl.figure(figstart+1)
    pl.clf()

    means_of_imsize = []
    stds_of_imsize = []
    for imsize in xrange(*imsizerange):
        print "Image Size = %i.  " % imsize,
        xoff,yoff,exoff,eyoff,means,stds = plot_montecarlo_test_offsets(xsh,
                            ysh, imsize, noise=noise, name="%i "%imsize,  **kwargs)
        means_of_imsize.append(means)
        stds_of_imsize.append(stds)

    imsizes = np.arange(*imsizerange)
    pl.figure(figstart+2)
    pl.clf()
    xmeans,ymeans,exmeans,eymeans = np.array(means_of_imsize).T
    xstds,ystds,exstds,eystds = np.array(stds_of_imsize).T
    pl.plot(imsizes,exmeans,label='$\\bar{\sigma_{x}}$')
    pl.plot(imsizes,eymeans,label='$\\bar{\sigma_{y}}$')
    pl.plot(imsizes,xstds,label='${\sigma_{x}(\mu)}$')
    pl.plot(imsizes,ystds,label='${\sigma_{y}(\mu)}$')

    pl.figure(figstart+3)
    pl.clf()
    pl.plot(imsizes,exmeans/xstds,label='$\\bar{\sigma_{x}} / \sigma_{x}(\mu)$')
    pl.plot(imsizes,eymeans/ystds,label='$\\bar{\sigma_{y}} / \sigma_{y}(\mu)$')

    return means_of_imsize,stds_of_imsize
