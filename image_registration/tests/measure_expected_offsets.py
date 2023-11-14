from .cross_correlation_shifts import cross_correlation_shifts
from .register_images import register_images
from . import chi2_shifts
from .fft_tools import dftups,upsample_image,shift,smooth
from .tests import registration_testing as rt

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
        print('%s took %0.5g s' % (func.__name__, (t2-t1)))
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
                for ii in range(nsamples)]

    xoff,yoff,exoff,eyoff = list(zip(*results))

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

def montecarlo_tests_of_imsize(xsh,ysh,imsizerange=[15,105,5],noise=0.5,
        figstart=0, clear=True, namepre="", **kwargs):
    """
    Perform many monte-carlo tests as a function of the image size
    """
    pl.figure(figstart+1)
    if clear: pl.clf()

    means_of_imsize = []
    stds_of_imsize = []
    for imsize in range(*imsizerange):
        print("Image Size = %i.  " % imsize, end=' ')
        xoff,yoff,exoff,eyoff,means,stds = plot_montecarlo_test_offsets(xsh,
                            ysh, imsize, noise=noise, name=namepre+"%i "%imsize,  **kwargs)
        means_of_imsize.append(means)
        stds_of_imsize.append(stds)

    imsizes = np.arange(*imsizerange)
    pl.figure(figstart+2)
    if clear: pl.clf()
    xmeans,ymeans,exmeans,eymeans = np.array(means_of_imsize).T
    xstds,ystds,exstds,eystds = np.array(stds_of_imsize).T
    pl.plot(imsizes,exmeans,label='$\\bar{\sigma_{x}}$')
    pl.plot(imsizes,eymeans,label='$\\bar{\sigma_{y}}$')
    pl.plot(imsizes,xstds,label='${\sigma_{x}(\mu)}$')
    pl.plot(imsizes,ystds,label='${\sigma_{y}(\mu)}$')
    pl.xlabel("Image Sizes")
    pl.ylabel("X and Y errors")
    pl.title("Noise Level = %f" % noise)

    pl.figure(figstart+3)
    if clear: pl.clf()
    pl.plot(imsizes,exmeans/xstds,label='$\\bar{\sigma_{x}} / \sigma_{x}(\mu)$')
    pl.plot(imsizes,eymeans/ystds,label='$\\bar{\sigma_{y}} / \sigma_{y}(\mu)$')
    pl.xlabel("Image Sizes")
    pl.ylabel("Ratio of measured to monte-carlo error")
    pl.title("Noise Level = %f" % noise)

    print("Ratio mean measure X error to monte-carlo X standard dev: ", np.mean(exmeans/xstds))
    print("Ratio mean measure Y error to monte-carlo Y standard dev: ", np.mean(eymeans/ystds))

    return np.array(means_of_imsize).T,np.array(stds_of_imsize).T

def monte_carlo_tests_of_noiselevel(xsh,ysh,noiselevels,imsize=25, figstart=0, clear=True,
        namepre="", **kwargs):
    pl.figure(figstart+1)
    if clear: pl.clf()

    means_of_noise = []
    stds_of_noise = []
    for noise in noiselevels:
        print("Noise Level = %f.  " % noise, end=' ')
        xoff,yoff,exoff,eyoff,means,stds = plot_montecarlo_test_offsets(xsh,
                            ysh, imsize, noise=noise, name=namepre+"%0.2f "%noise,  **kwargs)
        means_of_noise.append(means)
        stds_of_noise.append(stds)

    noises = noiselevels
    pl.figure(figstart+2)
    if clear: pl.clf()
    xmeans,ymeans,exmeans,eymeans = np.array(means_of_noise).T
    xstds,ystds,exstds,eystds = np.array(stds_of_noise).T
    pl.plot(noises,exmeans,label='$\\bar{\sigma_{x}}$')
    pl.plot(noises,eymeans,label='$\\bar{\sigma_{y}}$')
    pl.plot(noises,xstds,label='${\sigma_{x}(\mu)}$')
    pl.plot(noises,ystds,label='${\sigma_{y}(\mu)}$')
    pl.xlabel("Noise Levels")
    pl.ylabel("X and Y errors")
    pl.title("Image Size = %i" % imsize)

    pl.figure(figstart+3)
    if clear: pl.clf()
    pl.plot(noises,exmeans/xstds,label='$\\bar{\sigma_{x}} / \sigma_{x}(\mu)$')
    pl.plot(noises,eymeans/ystds,label='$\\bar{\sigma_{y}} / \sigma_{y}(\mu)$')
    pl.xlabel("Noise Levels")
    pl.ylabel("Ratio of measured to monte-carlo error")
    pl.title("Image Size = %i" % imsize)

    print("Ratio mean measure X error to monte-carlo X standard dev: ", np.mean(exmeans/xstds))
    print("Ratio mean measure Y error to monte-carlo Y standard dev: ", np.mean(eymeans/ystds))

    return np.array(means_of_noise).T,np.array(stds_of_noise).T

def centers_to_edges(arr):
    dx = arr[1]-arr[0]
    newarr = np.linspace(arr.min()-dx/2,arr.max()+dx/2,arr.size+1)
    return newarr

def monte_carlo_tests_of_both(xsh,ysh,noiselevels, imsizes, figstart=12, 
        clear=True, namepre="", **kwargs):
    pl.figure(figstart+1)
    pl.clf()

    pars = [[plot_montecarlo_test_offsets(xsh, ysh, imsize, noise=noise,
        name=namepre+"%0.2f "%noise,  **kwargs)
            for noise in noiselevels]
            for imsize in imsizes]

    means = np.array([[p[4] for p in a] for a in pars])
    stds = np.array([[p[5] for p in a] for a in pars])

    pl.subplot(221)
    pl.title("$\sigma_x$ means")
    pl.pcolormesh(centers_to_edges(noiselevels),centers_to_edges(imsizes),means[:,:,2])
    pl.subplot(222)
    pl.title("$\sigma_y$ means")
    pl.pcolormesh(centers_to_edges(noiselevels),centers_to_edges(imsizes),means[:,:,3])
    pl.subplot(223)
    pl.title("$\mu_x$ stds")
    pl.pcolormesh(centers_to_edges(noiselevels),centers_to_edges(imsizes),stds[:,:,0])
    pl.subplot(224)
    pl.title("$\mu_y$ stds")
    pl.pcolormesh(centers_to_edges(noiselevels),centers_to_edges(imsizes),stds[:,:,1])

    for ii in range(1,5):
        pl.subplot(2,2,ii)
        pl.xlabel("Noise Level")
        pl.ylabel("Image Size")
    
    return np.array(means),np.array(stds)


def perform_tests(nsamples=100):

    moi,soi = montecarlo_tests_of_imsize(3.7, -1.2, figstart=0,
            chi2_shift=chi2_shifts.chi2_shift,
            imsizerange=[15, 105, 5],nsamples=nsamples)
    moiB,soiB = montecarlo_tests_of_imsize(-9.1, 15.9, figstart=0, clear=False,
            chi2_shift=chi2_shifts.chi2_shift, imsizerange=[15,105,5],
            nsamples=nsamples)

    moi2,soi2 = montecarlo_tests_of_imsize(3.7, -1.2, figstart=3,
            chi2_shift=chi2_shifts.chi2_shift_iterzoom,
            imsizerange=[15, 105, 3],nsamples=nsamples)
    moi2B,soi2B = montecarlo_tests_of_imsize(-9.1, 15.9, figstart=3,
            clear=False, chi2_shift=chi2_shifts.chi2_shift_iterzoom,
            imsizerange=[15,105,3], nsamples=nsamples)

    mos,sos = monte_carlo_tests_of_noiselevel(3.7, -1.2, figstart=6,
            chi2_shift=chi2_shifts.chi2_shift,
            noiselevels=np.logspace(-1,1),nsamples=nsamples)
    mosB,sosB = monte_carlo_tests_of_noiselevel(-9.1, 15.9, figstart=6,
            clear=False, chi2_shift=chi2_shifts.chi2_shift,
            noiselevels=np.logspace(-1,1), nsamples=nsamples)

    mos2,sos2 = monte_carlo_tests_of_noiselevel(3.7, -1.2, figstart=9,
            chi2_shift=chi2_shifts.chi2_shift_iterzoom,
            noiselevels=np.logspace(-1,1), nsamples=nsamples)
    mos2B,sos2B = monte_carlo_tests_of_noiselevel(-9.1, 15.9, figstart=9,
            clear=False, chi2_shift=chi2_shifts.chi2_shift_iterzoom,
            noiselevels=np.logspace(-1,1), nsamples=nsamples)

    return locals()

def twod_tests(nsamples=100):
    mob,sob = monte_carlo_tests_of_both(3.7, -1.2, np.linspace(0.1,10,10),
            np.arange(15,105,5), figstart=12, clear=True, namepre="",
            chi2_shift=chi2_shifts.chi2_shift_iterzoom, nsamples=nsamples)
