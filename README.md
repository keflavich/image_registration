Image Registration Methods for Astronomy
========================================
(intended for *extended emission*, not stellar images!)

**Documentation:** http://image-registration.rtfd.org

### Usage

- `pip install image_registration` or 
- `git clone https://github.com/keflavich/image_registration`

```python
from image_registration import chi2_shift
from image_registration.fft_tools import shift
import image_registration

#Generate Sample Image
image = image_registration.tests.make_extended(100)
offset_image = image_registration.tests.make_offset_extended(image, 4.76666, -12.33333333333333333333333, noise=0.1)

#Get Fused Image
xoff, yoff, exoff, eyoff = chi2_shift(image, offset_image)
corrected_image2 = shift.shiftnd(offset_image, (-yoff, -xoff))
```

### Requirements:
Install the following version of the packages to replicate this repository:
- FITS_tools==0.2
- matplotlib==3.4.2
- astropy==4.2.1
  
For the following packages latest version should work:
- scipy
- pytest

To replicate in conda you can use environment.yml given in repository

### Acknowledgments:
- Borrows heavily from
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

- Replicates much of the functionality of 
http://www.astro.ucla.edu/~mperrin/IDL/sources/subreg.pro

Also implements 
http://solarmuri.ssl.berkeley.edu/~welsch/public/software/cross_cor_taylor.pro


<img src="https://upload.wikimedia.org/wikipedia/commons/0/0f/Zenodo_logo.jpg" alt="Zenodo" width="200" style="float:left">
<img src="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco/v1397185883/572a01e5ceae5baf6fd82328b810a566.png" alt="Bitdeli badge" width="200" style="float:left">

