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
#import PIL
#import pydicom.pixel_data_handlers.pillow_handler as pillow_handler
from pydicom.data import get_testdata_files
from heapq import merge

print(__doc__)

#filename = get_testdata_files('CT_small.dcm')[0]
filename = 'MRI_Images/MRI01.dcm'
#filename = 'MRI_Images/MR000001'

#pydicom.config.image_handlers = [pillow_handler]
dataset = pydicom.dcmread(filename)

 
def histogram(data):
    print()
    print("Creating histogram please wait. (Paciencia)")
    data = np.array(dataset.pixel_array)
    min_pixel = np.ndarray.min(data)
    max_pixel = np.ndarray.max(data)
    print("Smallest Image Pixel Value.....",min_pixel)
    print("Largest Image Pixel Value.....",max_pixel)
    
    
    histY = [0]*65536
    histX = [0]*65536

    for i in range(0,len(data),1):
        for j in range(0,len(data),1):
            index = data[i][j]  
            histY[index] = histY[index]+1

    for i in range(0,len(histX),1):
        histX[i] = i

    plt.xlabel('Values')
    plt.ylabel('Frecuency')
    plt.title('Histogram')
    plt.plot(histX,histY, 'k')
    plt.grid(True)
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
    
    
    #data = dataset.pixel_array
    
    histogram(dataset.pixel_array)
    

except ValueError:
    print("error: algun header no disponible",ValueError)






