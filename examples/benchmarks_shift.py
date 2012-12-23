"""
    imsize   map_coordinates    fourier_shift
        50          0.016211       0.00944495
        84         0.0397182        0.0161059
       118          0.077543        0.0443089
       153          0.132948         0.058187
       187          0.191808        0.0953341
       221          0.276543          0.12069
       255          0.357552         0.182863
       289          0.464547          0.26451
       324          0.622776         0.270612
       358          0.759015         0.713239
       392          0.943339         0.441262
       426           1.12885         0.976379
       461           1.58367          1.26116
       495           1.62482         0.824757
       529           1.83506          1.19455
       563           3.21001          2.82487
       597           2.64892          2.23473
       632           2.74313          2.21019
       666           3.07002          2.49054
       700           3.50138          1.59507

Fourier outperforms map_coordinates slightly.  It wraps, though, while
map_coordinates in general does not.
"""
import itertools
import timeit
import time

import numpy as np

timings = {'map_coordinates':[],
           'fourier_shift':[],
           'skimage':[],
           #'griddata_nearest':[],
           #'griddata_linear':[],
           #'griddata_cubic':[],
           }

imsizes = np.round(np.linspace(50,1024,20))
imsizes = np.round(np.linspace(50,700,20)) # just playing around with what's reasonable for my laptop

for imsize in imsizes:
    t0 = time.time()
    setup = """
    import numpy as np
    #im = np.random.randn({imsize},{imsize})
    yy,xx = np.indices([{imsize},{imsize}])
    im = np.exp(-((xx-{imsize}/2.)**2+(yy-{imsize}/2.)**2)/(2**2*2.))
    yr = np.arange({imsize})
    xr = np.arange({imsize})
    import image_registration.fft_tools.shift as fsh
    import image_registration.fft_tools.zoom as fzm
    import scipy.interpolate as si
    import scipy.ndimage as snd
    points = zip(xx.flat,yy.flat)
    imflat = im.ravel()
    import skimage.transform as skit
    skshift = skit.AffineTransform(translation=[0.5,0.5])
    """.replace("    ","").format(imsize=imsize)

    fshift_timer = timeit.Timer("ftest=fsh.shiftnd(im,(0.5,0.5))",
            setup=setup)

    # too slow!
    # interp2d_timer = timeit.Timer("itest=si.interp2d(xr,yr,im)(xr-0.5,yr-0.5)",
    #        setup=setup)

    mapcoord_timer = timeit.Timer("mtest=snd.map_coordinates(im,[yy-0.5,xx-0.5])",
            setup=setup)

    # not exactly right; doesn't do quite the same thing as the others...
    # but wow, I wish I'd figured out how to use this a week ago...
    skimage_timer = timeit.Timer("stest=skit.warp(im,skshift)",setup=setup)

    # all slopw
    #grid_timer_nearest = timeit.Timer("gtest=si.griddata(points,imflat,(xx-0.5,yy-0.5), method='nearest')",
    #        setup=setup)
    #grid_timer_linear = timeit.Timer("gtest=si.griddata(points,imflat,(xx-0.5,yy-0.5), method='linear')",
    #        setup=setup)
    #grid_timer_cubic = timeit.Timer("gtest=si.griddata(points,imflat,(xx-0.5,yy-0.5), method='cubic')",
    #        setup=setup)

    print "imsize %i fourier shift " % imsize,
    timings['fourier_shift'].append( np.min(fshift_timer.repeat(3,10)) )
    print "imsize %i map_coordinates shift " % imsize,
    timings['map_coordinates'].append( np.min(mapcoord_timer.repeat(3,10)) )
    print "imsize %i skimage shift " % imsize,
    timings['skimage'].append( np.min(skimage_timer.repeat(3,10)) )
    #timings['griddata_nearest'].append( np.min(grid_timer_nearest.repeat(3,10)) )
    #timings['griddata_linear'].append( np.min(grid_timer_linear.repeat(3,10)) )
    #timings['griddata_cubic'].append( np.min(grid_timer_cubic.repeat(3,10)) )
    print "imsize %i done, %f seconds" % (imsize,time.time()-t0)


print "%10s " % "imsize"," ".join(["%16s" % t for t in timings.keys()])
for ii,sz in enumerate(imsizes):
    print "%10i " % sz," ".join(["%16.6g" % t[ii] for t in timings.values()])


import scipy.optimize as scopt

def f(x,a,b,c):
    return c+a*x**b

pm,err = scopt.curve_fit(f,imsizes[2:],timings['map_coordinates'][2:])
pf,err = scopt.curve_fit(f,imsizes[2:],timings['fourier_shift'][2:])

import matplotlib.pyplot as pl
pl.clf()
pl.loglog(imsizes,timings['map_coordinates'],'+')
pl.loglog(imsizes,timings['fourier_shift'],'x')
pl.loglog(imsizes,f(imsizes,*pm))
pl.loglog(imsizes,f(imsizes,*pf))
