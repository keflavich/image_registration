import image_registration
from image_registration.fft_tools import *
from pylab import *

noise_taper = True
imsize=100
xsh=3.75
ysh=1.2
image = image_registration.tests.make_extended(imsize)
offset_image_taper = image_registration.tests.make_offset_extended(image, xsh, ysh, noise=0.5, noise_taper=True)
offset_image = image_registration.tests.make_offset_extended(image, xsh, ysh, noise=0.5, noise_taper=False)
noise = 0.5/image_registration.tests.edge_weight(imsize)

im1 = image
im2 = offset_image_taper

print "SCALAR"
xoff, yoff, exoff, eyoff, (x, y, c2a) = image_registration.chi2_shift(image,
        offset_image, 0.1, return_error=True, verbose=2,
        upsample_factor='auto', return_chi2array=True)
print "SCALAR error: ",xoff,yoff,exoff,eyoff
print
xoff, yoff, exoff, eyoff, (x, y, c2) = \
        image_registration.chi2_shift(image, offset_image_taper, noise,
            return_error=True, verbose=2, upsample_factor='auto',
            return_chi2array=True)
c2map,term1,term2,term3 = image_registration.chi2n_map(image,offset_image_taper,noise,return_all=True)

xoff3,yoff3,exoff3,eyoff3,(x3,y3,c3) = image_registration.chi2_shifts.chi2_shift_iterzoom(image, offset_image_taper, noise, return_chi2array=True, return_error=True,verbose=True,mindiff=0.1)
xoff4,yoff4,exoff4,eyoff4,(x4,y4,c4) = image_registration.chi2_shifts.chi2_shift_iterzoom(image, offset_image, 0.5, return_chi2array=True, return_error=True,verbose=True,mindiff=0.1)

c2mapA,term1A,term2A,term3A = image_registration.chi2n_map(image,offset_image,0.5,return_all=True)
print "TAPERED error: ",xoff,yoff,exoff,eyoff
print "TAPERED error absolute difference: ",abs(xoff-xsh),abs(yoff-ysh)

print "ITER version: "
print "SCALAR"
print "TAPERED error: ",xoff3,yoff3,exoff3,eyoff3
print "TAPERED error absolute difference: ",abs(xoff3-xsh),abs(yoff3-ysh)
print "TAPERED chi^2 min pos: ",x3.flat[c3.argmin()],y3.flat[c3.argmin()]
print "SCALAR error: ",xoff4,yoff4,exoff4,eyoff4
print "SCALAR error absolute difference: ",abs(xoff4-xsh),abs(yoff4-ysh)
print "SCALAR chi^2 min pos: ",x4.flat[c4.argmin()],y4.flat[c4.argmin()]

ylen,xlen = imsize,imsize
xcen = xlen/2-(1-xlen%2) 
ycen = ylen/2-(1-ylen%2) 

ymax = ycen - ysh
xmax = xcen - xsh

figure(1)
subplot(131); imshow(offset_image_taper); title('taper')
subplot(132); imshow(image); title("image")
subplot(133); imshow(noise)

figure(2)
pcolormesh(x,y,c2.real-c2.real.min()); colorbar()
contour(x,y,c2.real-c2.real.min(),levels=[0,2.3,6,8,100,200],cmap=cm.gray)
title("c2.real")
plot(-xsh,-ysh,'wx')

figure(3,figsize=[20,8])
#subplot(131); pcolormesh(x,y,term2ups.real); colorbar(); title('term2ups.real')
#contour(x,y,term2ups.real-term2ups.real.min(),levels=[0,2.3,6,8,100,200],cmap=cm.gray)
plot(-xsh,-ysh,'wx')
#subplot(132); pcolormesh(x,y,term3ups.real); colorbar(); title('term3ups.real')
plot(-xsh,-ysh,'wx')
subplot(133); pcolormesh(x,y,c2.real); colorbar(); title('c2.real')
contour(x,y,c2.real-c2.real.min(),levels=[0,2.3,6,8,100,200],cmap=cm.gray)
plot(-xsh,-ysh,'wx')

figure(4)
subplot(121)
imshow(term2); colorbar()
axlims = axis()
plot(xmax,ymax,'wx')
title("tapered term2")
axis(axlims)
subplot(122)
imshow(correlate2d(im1,im2))
title("tapered term2 - no error")
plot(xmax,ymax,'wx')
axis(axlims)
colorbar()

figure(5)
imshow(term3); colorbar()

title("term3")
figure(6)
imshow(c2map); colorbar()
axlims=axis()
title("c2map (tapered)")
plot(xmax,ymax,'wx')
plot(*np.unravel_index(c2map.argmin(),c2map.shape)[::-1],marker='+',color='w')
axis(axlims)

figure(7)
clf()
subplot(131)
imshow(term2); colorbar()
axlims=axis()
plot(xmax,ymax,'wx')
plot(*np.unravel_index(term2.argmin(),term2.shape)[::-1],marker='*',color='w',mec='w',mfc='none',mew=2)
title("term 2 tapered")
axis(axlims)
subplot(132)
imshow(term3); colorbar()
plot(xmax,ymax,'wx')
plot(*np.unravel_index(term2.argmin(),term2.shape)[::-1],marker='*',color='w',mec='w',mfc='none',mew=2)
plot(*np.unravel_index(c2map.argmin(),c2map.shape)[::-1],marker='+',color='w',mec='w',mfc='none',mew=2)
title("term 3 tapered")
axis(axlims)
subplot(133)
imshow(c2map); colorbar()
axlims=axis()
title("c2map (tapered)")
plot(xmax,ymax,'wx')
plot(*np.unravel_index(c2map.argmin(),c2map.shape)[::-1],marker='+',color='w')
axis(axlims)

axlims = (40,50,40,50)
figure(8)
clf()
subplot(131)
imshow(term2); colorbar()
plot(xmax,ymax,'wx')
plot(*np.unravel_index(term2.argmin(),term2.shape)[::-1],marker='*',color='w',mec='w',mfc='none',mew=2)
title("term 2 tapered")
axis(axlims)
subplot(132)
imshow(term3); colorbar()
plot(xmax,ymax,'wx')
plot(*np.unravel_index(term2.argmin(),term2.shape)[::-1],marker='*',color='w',mec='w',mfc='none',mew=2)
plot(*np.unravel_index(c2map.argmin(),c2map.shape)[::-1],marker='+',color='w',mec='w',mfc='none',mew=2)
title("term 3 tapered")
axis(axlims)
subplot(133)
imshow(c2map); colorbar()
title("c2map (tapered)")
plot(xmax,ymax,'wx')
plot(*np.unravel_index(c2map.argmin(),c2map.shape)[::-1],marker='+',color='w')
axis(axlims)

figure(9)
subplot(131); imshow(offset_image_taper/noise**2); colorbar(); title('taper/noise**2')
subplot(132); imshow(image**2/noise**2); colorbar(); title('image**2/noise**2')
subplot(133); imshow(image_registration.fft_tools.shift.shiftnd(image,(ysh,xsh))**2/noise**2); colorbar(); title('shift(image)**2/noise**2')
figure(10)
subplot(131); imshow(image); colorbar()
subplot(132); imshow(offset_image**2); colorbar()
subplot(133); imshow(noise**2); colorbar()
figure(11)
pcolormesh(x,y,c2a.real-c2a.real.min()); colorbar()
contour(x,y,c2a.real-c2a.real.min(),levels=[0,2.3,6,8,100,200],cmap=cm.gray)
plot(-xsh,-ysh,'wx')
title("Scalar upsample")
figure()
imshow(term2A); colorbar()
title("term2A")
plot(xmax,ymax,'wx')
figure()
print term1A,term3A
imshow(c2mapA); colorbar()
plot(xmax,ymax,'wx')
title("scalar")

im1 = image
im2 = offset_image
err = 1.
unups = fftn(im2/err**2)*np.conj(fftn(im1))
#figure()
#imshow(unups.real)
#title("fftn(im2/err**2)*np.conj(fftn(im1))")

figure(14)
pcolormesh(x3,y3,c3)
contour(x3,y3,c3-c3.min(),levels=[0,2.3,6.8,11,100,200],cmap=cm.gray)
axlims=axis()
errorbar(xoff3,yoff3,xerr=exoff3,yerr=eyoff3,color='w')
axis(axlims)
title("Iterative tapered")
figure(15)
pcolormesh(x4,y4,c4)
contour(x4,y4,c4-c4.min(),levels=[0,2.3,6.8,11,100,200],cmap=cm.gray)
axlims=axis()
errorbar(xoff4,yoff4,xerr=exoff4,yerr=eyoff4,color='w')
axis(axlims)
title("Iterative scalar")

