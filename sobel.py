import convolucion
import matplotlib.pyplot as plt
import numpy as np

class Kernel:
	
	def sobel(img):
		#kernel
		matrix_convolution1 = np.matrix("-1, 0, 1; -2, 0, 2; -1, 0, 1")
		matrix_convolution2 = np.matrix("-1, -2, -1; 0, 0, 0; 1, 2, 1")
		matrix1 = np.absolute(convolucion.convolution(img,matrix_convolution1))
		matrix2 = np.absolute(convolucion.convolution(img,matrix_convolution2))
		#para generar g suma de valores absolutos gx y gy
		return matrix1 + matrix2
		

	def gauss(img):
		matriz_convolucion = np.matrix("0, 1, 0; 1, -4, 1; 0, 1, 0")
		new_img = convolucion.convolution(img,matriz_convolucion)

	def histogram(data):
	    print()
	    print("Creating histogram please wait. (Paciencia)")
	    
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

#sobel = Sobel()