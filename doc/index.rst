.. image_registration documentation master file, created by
   sphinx-quickstart on Mon Sep  3 08:57:59 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Astronomical Image Registration
===============================

`Image Registration <https://github.com/keflavich/image_registration>`_

A toolkit for registering images of astronomical images containing primarily
extended flux (e.g., nebulae, radio and millimeter maps).

There are related packages scattered throughout the internet that do the same
thing, but with different features.

The general goal is to align images that look kind of like these:

.. image:: image.png
    :width: 400px
    :alt: The input image

.. image:: image_shifted_corrupted.png
    :width: 400px
    :alt: The input image shifted and corrupted with gaussian noise

Module APIs:
------------

   :doc:`image_registration` Module

   :doc:`image_registration.fft_tools` Module

   :doc:`image_registration.tests` Module


   The most successful of the methods implemented here is :func:`~image_registration.chi2_shift`.

   There is an ipython notebook demonstration of the code `here <https://github.com/keflavich/image_registration/blob/master/doc/CrossCorrelationSimulation.pdf?raw=true>`_

.. autofunction:: chi2_shift

.. currentmodule:: image_registration



Related Programs
----------------
    `Varosi + Landsman astrolib correl_optimize <http://idlastro.gsfc.nasa.gov/ftp/pro/image/correl_optimize.pro>`_ :
        Uses cross-correlation with "reduction" and "magnification" factors for
        speed and accuracy respectively; this method is relatively slow when
        using the complete information in the image (the magnification process
        increases the size of the image directly)

    `Brian Welsch's cross-cor taylor <http://solarmuri.ssl.berkeley.edu/~welsch/public/software/cross_cor_taylor.pro>`_ :
        Uses the cross-correlation peak to measure the pixel peak of the
        offset, then does a 2nd order taylor-expansion around that peak to
        achieve sub-pixel accuracy.  Is fast and generally quite accurate, but
        can be subject to bias.

    `Manuel Guizar's Efficient Sub-Pixel Registration <http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation>`_ :
        A matlab version of the main method implemented in this code.  Is fast
        and accurate.  The speed comes from making use of the fourier zoom /
        fourier scaling property.

    `Marshall Perrin's sub-pixel image registration <http://www.astro.ucla.edu/~mperrin/IDL/sources/subreg.pro>`_ :
        Implements many cross-correlation based methods, with sub-pixel
        registration based off of centroiding,  gaussian fitting, and many
        variations thereupon.  The gaussian approach is also implemented here,
        but is highly biased and inaccurate in general.  As a sidenote, I tried
        using the "gaussian fit after high-pass filter" approach, but found
        that it really didn't work - it helped remove SOME of the large-scale
        junk, but it didn't end up changing the shape of the peak in a helpful
        way.


.. :mod:`image_registration` Module
.. --------------------------------

.. automodule:: image_registration
    :members:
    :undoc-members:

.. :mod:`image_registration.fft_tools` Module
.. ------------------------------------------

.. automodule:: image_registration.fft_tools
    :members:
    :undoc-members:

.. :mod:`image_registration.tests` Module
.. --------------------------------------

.. automodule:: image_registration.tests
    :members:
    :undoc-members:

.. Contents:
.. ~~~~~~~~~

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`image_registration.FITS_tools`
* :doc:`image_registration.fft_tools`
* :doc:`image_registration`
* :doc:`image_registration.tests`
  

