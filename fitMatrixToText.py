#Author: Adriel Kim
#A hacky program for getting the image data from a .fits file
#Putting the array of pixel values in a text file, which can be read by a Cuda C program
#Because I don't think i'll be able to get Python Numba to work.(I wish I had my own GPU ;| )
#Ask to get numba on department servers if it turns out to be a viable tool

import astropy.io.fits as pyfits
import numpy as np

def convertFitToText(fits_name, layer, fileName):
    #Note: why is the length of bias fits only 1?
    #Maybe they represent layers within image (ask about that)

    fits_file = pyfits.open(fits_name)

    img1= fits_file[layer].data

    a_file = open(fileName, "w")
    a_file.write(str(img1.shape[0])+","+str(img1.shape[1])+"\n")
    for row in img1:
        np.savetxt(a_file, row)
    a_file.close()

fits_list = ["BIAS-2013-11-03-000.fits", "HSCA90333426.fits"]
convertFitToText(fits_list[1], 1, 'imgraw1.txt')


