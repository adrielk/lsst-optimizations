#Bias correction using astropy
#Takes raw and bias, performs bias correction, writes resulting image
import astropy.io.fits as pyfits
import numpy as np

fits = pyfits.open('HSCA90333426.fits')
fitsBias = pyfits.open('BIAS-2013-11-03-000.fits')
data = fits[0]
data2 = fitsBias[0]
image = fits[1].data
biasImage = fitsBias[0].data
#print(data.header.keys)
image = image[0:4176,0:2048]#making images same size...(hacky i know. I don't even know if these images correspond to each other)
print(image.shape)
print(biasImage.shape)
biasCorr = image - biasImage
fits[1].data = biasCorr

hdu = pyfits.PrimaryHDU(biasCorr)

hdu.writeto('biasCorrected3.fits')

