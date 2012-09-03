#!/usr/bin/env python

from distutils.core import setup

with open('README') as file:
    long_description = file.read()

with open('CHANGES') as file:
    long_description += file.read()

# no versions yet from agpy import __version__ as version

setup(name='image_registration',
      version='0.1',
      description='Image Registration Tools for extended images in astronomy.',
      long_description=long_description,
      author='Adam Ginsburg',
      author_email='adam.g.ginsburg@gmail.com',
      url='https://github.com/keflavich/image_registration',
      packages=['image_registration', 'image_registration/fft_tools',
          'image_registration/tests'], 
     )
