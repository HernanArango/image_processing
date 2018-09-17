"""
=======================================
Read DICOM and ploting using matplotlib
=======================================
This example illustrates how to open a DICOM file, print some dataset
information, and show it using matplotlib.
"""

# authors : Guillaume Lemaitre <g.lemaitre58@gmail.com>
# license : MIT

import matplotlib.pyplot as plt
import pydicom
import numpy as np
import PIL
import pydicom.pixel_data_handlers.pillow_handler as pillow_handler
from pydicom.data import get_testdata_files


print(__doc__)

#filename = get_testdata_files('CT_small.dcm')[0]
#filename = 'MRI_Images/MRI01.dcm'
filename = 'MRI_Images/MR000001'

pydicom.config.image_handlers = [pillow_handler]
dataset = pydicom.dcmread(filename)

def histogram(data):
    print("Creating histogram please wait. (Paciencia)")
    plt.figure() # create a new figure
    plt.title('Histogram')
    plt.hist(data) # plot a histogram of the pixel values
    plt.show()

# Normal mode:
try:
    print()
    print("Filename.........:", filename)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Protocol Name.......:", dataset.ProtocolName)
    #print("Manufacter.......:", dataset.Manufacturer)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)
    print("SeriesNumber.......:", dataset.SeriesNumber)
    print("Spacing Between Slices.......:", dataset.SeriesNumber)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

    
    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))
    #print("Smallest Image Pixel Value.....", dataset.SmallestImagePixelValue)
    #print("Largest Image Pixel Value.....", dataset.LargestImagePixelValue) 

    
    # plot the image using matplotlib
    #plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    #plt.show()
    
    

    data = np.array(dataset.pixel_array)
    print("Smallest Image Pixel Value.....",np.ndarray.min(data))
    print("Largest Image Pixel Value.....",np.ndarray.max(data))
    histogram(data)
    

except ValueError:
    print("error: algun header no disponible",ValueError)





