Fourier Image Manipulation Tools
================================

There are a few interesting Fourier-based image manipulation tools implemented
in this package.

Shift Theorem
-------------

The Fourier shift theorem allows an image to be shifted in any directions by an
arbitrary amount, including sub-pixel shifts.  It is more computationally
efficient than most interpolation techniques (or at least, so I think).  It is
described well in `this lecture <http://www.cs.unm.edu/~williams/cs530/theorems6.pdf>`_ 
and in `the NRAO interferometry course <http://www.cv.nrao.edu/course/astr534/FourierTransforms.html>`_.

.. math::
    FT[f(t-t_0)](x) = e^{-2 \pi i x t_0} F(x)

The shift code is in :mod:`image_registration.fft_tools.shift`.

Similarity Theorem
------------------
The similarity theorem, or scale theorem, allows you to upsample timestreams.
You cannot gain information below the image scale, but it is useful for getting
sub-pixel information about gaussian peaks, for example.  
`This NRAO lecture <http://www.cv.nrao.edu/course/astr534/FTSimilarity.html>`_
details the math.

.. math::
    {f(ax)\Leftrightarrow
    \frac{F\left(s/a\right)}{\left|a\right|}}

The zoom and upsample methods are in :mod:`image_registration.fft_tools.scale` 
and :mod:`image_registration.fft_tools.zoom`.

Resources
---------
I made use of a lot of not particularly easy to find documents when writing this code.

 * `Image Resampling by Neil Dodgson <http://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-261.pdf>`_ is a book
 * `Comparison of Interpolation Methods for Image Resampling <http://www.cs.uic.edu/~kenyon/Papers/Comparison.of.Interpolating.Methods.Parker.Kenyon.Troxel.pdf>`_
 * `The Similarity Theorem <http://www.technick.net/public/code/cp_dpage.php?aiocp_dp=guide_dft_appendix_d_similarity>`_

As stated elsewhere, though, the main inspiration for this work was `Manuel Guizar's DFT upsampling technique
<http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation>`_
