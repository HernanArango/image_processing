from tools.convolucion import *
import matplotlib.pyplot as plt
import numpy as np


#OTSU 
def otsu(hist):
	#trabajamos solo con los valores de Y (frecuencias)
	hist = hist[1]
	total_pixel = len(hist)

	q1 = 0
	q2 = 0
	u1 = 0
	u2 = 0
	Q1 = 0
	Q2 = 0

	for i in range(0,total_pixel):
	  #q(t) = sum p(i)
	  q1 += hist[i]

	q2 = total_pixel - q1

	q1 = int(q1 / total_pixel)
	q2 = q2

	for i in range(0,total_pixel):
	  # i * P(i)  
	  u1 += (i * hist[i]) / q1
	  u2 += (i * hist[i]) / q2
	
	u1 = int(u1)
	u2 = int(u2)

	for i in range(0,total_pixel):
	  Q1 += int(hist[i]/u1)*(i - q1)^2
	  Q2 += int(hist[i]/u2)*(i - q2)^2
	
	Qw = (q1*Q1) + (q2*Q2)

	print("q1",q1)
	print("q2",q2)
	print("u1",u1)
	print("u2",u2)

	

	print("u1",u1)
	print("u2",u2)

	Q = Qw + ((q1 * q2) * (u1 - u2)^2)
	
	return Q

def sobel(img):
	#kernel
	
	matrix_convolution1 = np.matrix("-1, 0, 1; -2, 0, 2; -1, 0, 1")
	matrix_convolution2 = np.matrix("-1, -2, -1; 0, 0, 0; 1, 2, 1")
	matrix1 = np.absolute(convolution(img,matrix_convolution1))
	matrix2 = np.absolute(convolution(img,matrix_convolution2))
	#para generar g suma de valores absolutos gx y gy
	g = matrix1 + matrix2
	
	print("otsu")
	hist = histogram(g.astype(int))
	print(hist)
	#calculanting umbral
	umbral = otsu(hist)

	print("umbral",umbral)
	umbral = 10000
	rows, cols = img.shape
	#creating binary matrix
	for i in range(0,rows):
		for j in range(0,cols):
			if g[i][j]  >= umbral:
				g[i][j] = 0
			else:
				g[i][j] = 1
	
	return g

def gauss(img):
	matriz_convolucion = np.matrix("0, 1, 0; 1, -4, 1; 0, 1, 0")
	new_img = convolucion.convolution(img,matriz_convolucion)
	return new_img


def histogram(data):
    
    print("Creating histogram please wait. (Paciencia)")
    
    min_pixel = np.ndarray.min(data)
    max_pixel = int(np.ndarray.max(data))
    print("Smallest Image Pixel Value.....",min_pixel)
    print("Largest Image Pixel Value.....",max_pixel)
    print (max_pixel+1)
    
    histY = [0] * (max_pixel+1)
    histX = [0] * (max_pixel+1)

    for i in range(0,len(data),1):
    	#histX[i] = i
    	for j in range(0,len(data),1):
            index = data[i][j]  
            histY[index] = histY[index]+1
    
    for i in range(0,len(histX),1):
    	histX[i] = i
    
    """
    plt.xlabel('Values')
    plt.ylabel('Frecuency')
    plt.title('Histogram')
    plt.plot(histX,histY, 'k')
    plt.grid(True)
    plt.show()
	"""
    return [histX,histY]