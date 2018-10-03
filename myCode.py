import pydicom
from matplotlib import pyplot as plt
import numpy as np


ds = pydicom.dcmread("./MRI_Images/MRI01.dcm")

rows = ds.Rows
columns = ds.Columns
seriesn = ds.SeriesNumber
instancenum = ds.InstanceNumber

print(rows)
print(columns)
print(seriesn)
print(instancenum)

pixArray = ds.pixel_array
print(pixel_array)

#histx: histogram x values
try:
	histY = []*ds.LargestImagePixelValue
except:
	histY = [0]*65536

#print(pixArray[0][0])
#print(histX[index])
histX = [0]*65536

for i in range(0,len(pixArray),1):
	for j in range(0,len(pixArray[i]),1):
		index = pixArray[i][j]	
		histY[index] = histY[index]+1

for i in range(0,len(histX),1):
	histX[i] = i

plt.xlabel('Values')
plt.ylabel('Frecuency')
plt.title('Histogram')
#plt.bar(histX, histY, width = 1)
#plt.hist(x, 50,density=True)
plt.plot(histX,histY, 'k')
plt.grid(True)
plt.show()