from tools.convolucion import *
import matplotlib.pyplot as plt
import numpy as np


#OTSU 
def otsu(hist):
	#trabajamos solo con los valores de Y (frecuencias)
	hist = hist[1]
	total_pixel = len(hist)
	total_value = 0
	print("total_pixel",total_pixel)
	
	q1 = 0
	q2 = 0
	u1 = 0
	u2 = 0
	w = 0
	Q = 0
	total_sum_pixel = 0
	threshold = 0

	for i in range(0,total_pixel):
		total_sum_pixel += hist[i] * i
		total_value += hist[i]


	for i in range(0,total_pixel):

		
		q1 += hist[i]
		#q2 = total_value - q1
		q2 = total_pixel - q1

		if (q1 == 0 or q2==0):
			continue
		
		w += i * hist[i]
		# u1 = sum (i*p(i))/q1(t)
		u1 += w / q1

		if (u1 == 0):
			continue

		u2 = (total_sum_pixel - w ) / q2


		Q_temp = ((q1 * q2) * ((u1 - u2) * (u1 - u2)))
		
		if (Q < Q_temp):
			Q = Q_temp
			threshold = i

	return threshold



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
	#print(hist)
	#calculanting umbral
	umbral = otsu(hist)

	print("umbral",umbral)
	#umbral = 10000
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