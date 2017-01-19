fft_tools Package
=================

:mod:`fft_tools` Package
------------------------

.. automodapi:: image_registration.fft_tools

:mod:`convolve_nd` Module
-------------------------

.. automodapi:: image_registration.fft_tools.convolve_nd

:mod:`correlate2d` Module
-------------------------

.. automodapi:: image_registration.fft_tools.correlate2d

:mod:`fast_ffts` Module
-----------------------

.. automodapi:: image_registration.fft_tools.fast_ffts

:mod:`shift` Module
-------------------

.. automodapi:: image_registration.fft_tools.shift

:mod:`zoom` Module
-------------------

.. automodapi:: image_registration.fft_tools.zoom

:mod:`scale` Module
-------------------

.. automodapi:: image_registration.fft_tools.scale

:mod:`upsample` Module
----------------------

Fourier upsampling (or interpolation, scaling, zooming) is achieved via DFTs
using a dot product rather than the usual fft, as there is (probably?) no way
to perform FFTs with a different kernel.

`This notebook <http://nbviewer.ipython.org/urls/raw.github.com/keflavich/image_registration/master/doc/Fourier%2520Scaling%2520%3D%2520Zooming%2520%3D%2520Similarity.ipynb>`_
demonstrates 1-d Fourier upsampling.

.. automodapi:: image_registration.fft_tools.upsample

