# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .cross_correlation_shifts import cross_correlation_shifts, cross_correlation_shifts_FITS
    from .chi2_shifts import chi2_shift, chi2n_map, chi2_shift_iterzoom
    from .register_images import register_images
    from . import fft_tools
    from . import tests
    from . import FITS_tools
