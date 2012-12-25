"""
A demonstration intended to illustrate possible problems with the Gaussian
fitting approach to measuring the cross-correlation peak.

If the cross-correlation peak is well-represented by a single Gaussian
component, the errors acquired from the normal least-squares fit should be
representative of the true error in the measurement.  However, there is
frequently additional structure in the images such that they are not perfectly
represented by a single gaussian component.
"""
try:
    import scipy.optimize as scopt
    import matplotlib.pyplot as pl
    import numpy as np

    def g(x,dx,s):
        return np.exp(-(x-dx)**2/(2.*s**2))

    x = np.linspace(-10,10,1000)

    for ii,(dx,s1,s2,scale1) in enumerate([(0.7,5,0.2,1),(0.7,9,0.6,1),(0.7,5,0.2,0.25)]):
        pl.figure(ii+1)
        pl.clf()
        twocomp = g(x,0.0,s1)*scale1+g(x,dx,s2)
        pl.title("dx=%0.1f $\sigma_1=%i$ $\sigma_2=%0.1f$" % (dx,s1,s2))
        pl.plot(x,twocomp,label='Two-component example')
        pars,err = scopt.curve_fit(g,x,twocomp)
        pl.plot(x,g(x,*pars),label='Best-fit gaussian')
        pl.legend(loc='best')
except ImportError:
    # this is just a demo; failure to import scipy should not break the build
    pass
