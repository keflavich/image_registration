collect_ignore = ["setup.py", "debug_test.py",
        "image_registration/tests/debug_test.py", "build",
        "ignore","*ipython*",
        "doc/sphinxext",
        "doc/_build",
        "doc/_static",
        "doc/_templates",
        "doc/conf.py",
        "examples/benchmarks_shift.py", # too slow for tests
        "examples/benchmarks_zoom.py", # too slow for tests
        ".git"
        ]

import warnings
warnings.filterwarnings("ignore",
    category=RuntimeWarning)

#    WARNING: RuntimeWarning: invalid value encountered in divide [image_registration.fft_tools.convolve_nd]
