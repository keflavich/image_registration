"""
Benchmark tests comparing fourier zoom to other techniques

    imsize   map_coordinates     fourier_zoom
        50         0.0523551        0.0214219
       101          0.214316         0.092334
       153          0.499386         0.245633
       204           1.13154         0.506588
       255           2.03562          1.33743
       306            4.2135          3.11828
       358           6.59853          5.83727
       409           8.52711          9.29976
       460           11.1365          12.3836
       511           17.8496          17.9716
       563           19.4098          24.3088
       614            10.559          29.7606
       665           11.6219          15.7843
       716           15.1977          20.9938
       768           16.6483          22.0162
       819           19.7834          28.2969
       870           21.6399          34.0512
       921           24.7352          41.2564
       973           27.3512          48.8034
      1024           32.1365          51.9153

Somewhat disappointing, but the fourier approach can be parallelized pretty
easily, I don't know about map_coordinates.

Also, I'm more than a little suspicious of these results; the execution times
took a precipitous drop between 615 and 665 seconds, which I think is an
indication that I was running too much other junk in parallel.

It's also pretty weird that the fourier approach seems to go up more steeply
than the map_coordinates approach; it could indicate memory limitations on my
machine.

Another test:
    imsize   map_coordinates     fourier_zoom     skimage_zoom
        50          0.086158        0.0248492        0.0535769
       100          0.308121         0.114798         0.199375
       150          0.762103         0.300909          0.53565
       200           1.32903         0.753605         0.900321
       250           2.13581          1.33399          1.39549
       300           2.98052          2.05711          2.18081
       350            4.3403          3.49586          2.69526
       400           5.45967          4.82973          3.68139
       450           7.97855           6.9673          4.62726
       500             8.367          8.79732          3.99151
    
skimage is the clear winner, except for small images.  Hrmph.

"""
import itertools
import timeit
import time

import numpy as np

zoomtimings = {'map_coordinates':[],
           'fourier_zoom':[],
           'skimage_zoom':[],
           #'griddata_nearest':[],
           #'griddata_linear':[],
           #'griddata_cubic':[],
           }

imsizes = np.round(np.linspace(50,1024,20))
imsizes = np.round(np.linspace(50,500,10))

for imsize in imsizes:
    t0 = time.time()
    setup = """
    import numpy as np
    #im = np.random.randn({imsize},{imsize})
    yy,xx = np.indices([{imsize},{imsize}])
    im = np.exp(-((xx-{imsize}/2.)**2+(yy-{imsize}/2.)**2)/(2**2*2.))
    # upsample by factor of 2
    yr = np.linspace(0,(({imsize}*2)-1)/2.,{imsize}*2)-0.25 # middle conventions...
    xr = np.linspace(0,(({imsize}*2)-1)/2.,{imsize}*2)-0.25 # middle conventions...
    xxnew,yynew = np.meshgrid(xr,yr)
    import image_registration.fft_tools.zoom as fzm
    import scipy.interpolate as si
    import scipy.ndimage as snd
    points = zip(xx.flat,yy.flat)
    imflat = im.ravel()
    import skimage.transform as skit
    """.replace("    ","").format(imsize=imsize)

    fzoom_timer = timeit.Timer("ftest=fzm.zoomnd(im,usfac=2,outshape=xxnew.shape)",
            setup=setup)

    # too slow!
    # interp2d_timer = timeit.Timer("itest=si.interp2d(xr,yr,im)(xr-0.5,yr-0.5)",
    #        setup=setup)

    mapzoom_timer = timeit.Timer("mtest=snd.map_coordinates(im,[yynew,xxnew])",
            setup=setup)

    skimagezoom_timer = timeit.Timer("stest=skit.resize(im,xxnew.shape)",setup=setup)

    # all slopw
    #grid_timer_nearest = timeit.Timer("gtest=si.griddata(points,imflat,(xx-0.5,yy-0.5), method='nearest')",
    #        setup=setup)
    #grid_timer_linear = timeit.Timer("gtest=si.griddata(points,imflat,(xx-0.5,yy-0.5), method='linear')",
    #        setup=setup)
    #grid_timer_cubic = timeit.Timer("gtest=si.griddata(points,imflat,(xx-0.5,yy-0.5), method='cubic')",
    #        setup=setup)

    print ("imsize %i fourier zoom " % imsize,
    zoomtimings['fourier_zoom'].append( np.min(fzoom_timer.repeat(3,10)) ))
    print ("imsize %i map_coordinates zoom " % imsize,
    zoomtimings['map_coordinates'].append( np.min(mapzoom_timer.repeat(3,10)) ))
    print ("imsize %i skimage zoom " % imsize,
    zoomtimings['skimage_zoom'].append( np.min(skimagezoom_timer.repeat(3,10)) ))
    #zoomtimings['griddata_nearest'].append( np.min(grid_timer_nearest.repeat(3,10)) )
    #zoomtimings['griddata_linear'].append( np.min(grid_timer_linear.repeat(3,10)) )
    #zoomtimings['griddata_cubic'].append( np.min(grid_timer_cubic.repeat(3,10)) )
    print ("imsize %i done, %f seconds" % (imsize,time.time()-t0))

print ("%10s " % "imsize"," ".join(["%16s" % t for t in zoomtimings.keys()]))
for ii,sz in enumerate(imsizes):
    print ("%10i " % sz," ".join(["%16.6g" % t[ii] for t in zoomtimings.values()]))


import scipy.optimize as scopt

def f(x,a,b):
    return a*x**b

pm,err = scopt.curve_fit(f,imsizes[1:],zoomtimings['map_coordinates'][1:])
pf,err = scopt.curve_fit(f,imsizes[1:],zoomtimings['fourier_zoom'][1:])
ps,err = scopt.curve_fit(f,imsizes[1:],zoomtimings['skimage_zoom'][1:])

import matplotlib.pyplot as pl
pl.clf()
pl.loglog(imsizes,zoomtimings['map_coordinates'],'+')
pl.loglog(imsizes,zoomtimings['fourier_zoom'],'x')
pl.loglog(imsizes,zoomtimings['skimage_zoom'],'x')
pl.loglog(imsizes,f(imsizes,*pm))
pl.loglog(imsizes,f(imsizes,*pf))
pl.loglog(imsizes,f(imsizes,*ps))
