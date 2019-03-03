import convolucion
import matplotlib.pyplot as plt
import numpy as np

class Kernel():
	
	def __init__(self):
		self.prueba="sdsd"

	#OTSU 
	def otsu(self,hist):
		#trabajamos solo con los valores de Y (frecuencias)
		hist = hist[1]
		total_pixel = len(hist)

		q1 = 0
		q2 = 0
		u1 = 0
		u2 = 0

		for i in hist:
		  #q(t) = sum p(i)
		  q1 += i

		q2 = total_pixel - q1

		for i in range(0,total_pixel):
		  # i * P(i)  
		  u1 += (i * hist[i]) / q1
		  u2 += (i * hist[i]) / q2
		"""
		for i in range(0,total_pixel):
		  Q1 += (hist[0][i]/u1)*(i - q1)^2
		  Q2 += (hist[0][i]/u2)*(i - q2)^2
		"""

		Q = (q1 * q2) * (u1 - u2)^2
		
		return Q

	def sobel(self,img):
		#kernel
		matrix_convolution1 = np.matrix("-1, 0, 1; -2, 0, 2; -1, 0, 1")
		matrix_convolution2 = np.matrix("-1, -2, -1; 0, 0, 0; 1, 2, 1")
		
		matrix1 = np.absolute(convolucion.convolution(img,matrix_convolution1))
		matrix2 = np.absolute(convolucion.convolution(img,matrix_convolution2))
		#para generar g suma de valores absolutos gx y gy
		g = matrix1 + matrix2
		
		print("otsu")
		#calculanting umbral
		umbral = self.otsu(self.histogram(g))

		rows, cols = img.shape
		#creating binary matrix
		for i in xrange(0,rows):
			for j in xrange(0,cols):
				if g[i][j]  >= umbral:
					g[i][j] = 1
				else:
					g[i][j] = 0
		
		return g

	def gauss(self,img):
		matriz_convolucion = np.matrix("0, 1, 0; 1, -4, 1; 0, 1, 0")
		new_img = convolucion.convolution(img,matriz_convolucion)


	def histogram(self,data):
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

	    return [histX,histY]
