#Displaying exposure using lsst pipelines

import lsst.daf.persistence as dafPersist
import lsst.afw.display as afwDisplay
butler = dafPersist.Butler(inputs='/home/adriel/DATA/rerun/processCcdOutputs')
data = butler.queryMetadata('calexp', ['visit', 'ccd'], dataId={'filter': 'HSC-R'})
print(data)
calexp = butler.get('calexp',dataId = {'filter':'HSC-R', 'visit':903334, 'ccd':23})
display = afwDisplay.getDisplay(backend = 'ds9')
display.mtv(calexp)

"""
src = butler.get('src', dataId = {'filter':'HSC-R', 'visit':903334, 'ccd':23})
print(len(src))
#print(src.getSchema())
print(src.schema.find("calib_psf_used"))

mask = calexp.getMask()
for maskName, maskBit in mask.getMaskPlaneDict().items():
    print('{}: {}'.format(maskName, display.getMaskPlaneColor(maskName)))
"""