from tools.convolucion import *
import matplotlib.pyplot as plt
import math
import numpy as np


#OTSU 
def otsu(img):
	hist = histogram(img)
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

	rows, cols = img.shape
	new_matriz = np.zeros((rows, cols))
	#creating binary matrix
	for i in range(0,rows):
		for j in range(0,cols):
			if img[i][j]  >= threshold:
				new_matriz[i][j] = 0
			else:
				new_matriz[i][j] = 1
	
	
	return new_matriz



def sobel(img):
	#kernel
	
	matrix_convolution1 = np.matrix("-1, 0, 1; -2, 0, 2; -1, 0, 1")
	matrix_convolution2 = np.matrix("-1, -2, -1; 0, 0, 0; 1, 2, 1")
	matrix1 = np.absolute(convolution(img,matrix_convolution1))
	matrix2 = np.absolute(convolution(img,matrix_convolution2))
	#para generar g suma de valores absolutos gx y gy
	g = matrix1 + matrix2
	
	print("otsu")
	"""
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
	"""
	new_img = otsu(g.astype(int))
	return new_img


def gauss(img,options):
	#matriz_convolucion = np.matrix("0, 1, 0; 1, -4, 1; 0, 1, 0")

	matriz_convolution = get_gaussian_filter(options)

	new_img = convolution(img,matriz_convolution)
	return new_img


def raleight(img,options):
	matriz_convolution = get_rayleigh_filter(options)
	return convolution(img,matriz_convolution)


def get_gaussian_filter(options):
	neighbours = int(options[0])
	sigma = int(options[1])
	'''
	This function calculates a NxN gaussian filter; N = neighbours * 2 + 1
	----------
	G(X,Y) = (1/ 2 * PI * SIGMA) * e^(-(X^2 + Y^2) / 2 * SIGMA^2)
	----------
	The values of X and Y are the difference between the current pixel and the center of the kernel
	For a 3x3 kernel, we have the following values of X and Y paired as tuples.
	----------
	[
	    [(-1,-1), (0, -1) (1, -1)],
	    [(-1, 0), (0, 0), (1, 0)],
	    [(-1, 1), (0, 1), (1, 1)]
	]
	'''
	N =  neighbours * 2 + 1

	#First, we create an array of shape NxN
	X = Y = np.zeros((N, N))

	#Then, we populate all the columns using np.arange - neighbours, to get the distance
	#between the element in the middle (index = neigboors) and the index given by arange

	#NOTE: If you want to change the indexes, similar to the Rayleigh function, change to the following line:
	#Y[:] = np.arange(N, dtype=np.float32)
	Y[:] = np.arange(N, dtype=np.float32) - neighbours

	#To calculate the rows, we transpose the columns
	X = np.transpose(Y)

	#NOTE: We do this to use numpy's vectorization, since we can have X and Y index values separated. 
	#Remember that the Gaussian filter in 2D is equal to multiplying two 1D Gaussians

	gaussian_filter = np.zeros((N, N))

	#We write down the left side (scalar value) of the equation: (1/ 2 * PI * SIGMA)
	left_side = (1 / (np.pi * 2 * np.power(sigma, 2))) 

	#We start writting down the right side: -(X^2 + Y^2) / 2 * SIGMA^2
	aux = -(np.power(X, 2) + np.power(Y, 2)) / (2 * np.power(sigma, 2))
	#Finally, we use the Exp function to calculate the e^(aux)
	right_side = np.exp(aux)

	#Multiply both sides of the equation
	gaussian_filter = left_side * right_side

	#To calculate the scalar value to multiply by, we sum over all the axes of the gaussian matrix, using np.sum
	factor_value = 1 / np.sum(gaussian_filter)

	return gaussian_filter#, factor_value

def get_rayleigh_filter(options):
	
	neighbours = int(options[0])
	sigma = int(options[1])

	'''
	This function calculates a NxN rayleigh filter; N = neighbours * 2 + 1
	----------
	G(X,Y) = [X * Y / SIGMA^4] * [e^(-(X^2 + Y^2) / 2 * SIGMA^2))]
	----------
	The values of X and Y are the indixes of each pixel
	For a 3x3 kernel, we have the following values of X and Y paired as tuples.
	----------
	[
	    [(0,0), (0, 1) (0, 2)],
	    [(1, 0), (1, 1), (1, 2)],
	    [(2, 0), (2, 1), (2, 2)]
	]
	'''
	N =  neighbours * 2 + 1

	#First, we create an array of shape NxN
	X = Y = np.zeros((N, N))

	#Then, we populate all the columns using np.arange to get the indexes
	Y[:] = np.arange(N, dtype=np.float32)

	#To calculate the rows, we transpose the columns
	X = np.transpose(Y)

	#NOTE: We do this to use numpy's vectorization, since we can have X and Y index values separated. 
	#Similar to gaussian 2D Rayleighs are equal to multiplying two 1D Rayleighs
	rayleigh_filter = np.zeros((N, N))

	#We write down the left side of the equation: [X * Y / SIGMA^4]
	left_side = np.divide(X * Y, np.power(sigma, 4))

	#We start writting down the right side: -(X^2 + Y^2) / 2 * SIGMA^2
	aux = -(np.power(X, 2) + np.power(Y, 2)) / (2 * np.power(sigma, 2))
	#Finally, we use the Exp function to calculate the e^(aux)
	right_side = np.exp(aux)


	#Multiply both sides of the equation
	rayleigh_filter = left_side * right_side

	#To calculate the scalar value to multiply by, we sum over all the axes of the gaussian matrix, using np.sum
	factor_value = 1 / np.sum(rayleigh_filter)

	return rayleigh_filter #, factor_value

def mediana(img):

    rows, cols = img.shape
    
    new_matriz = np.zeros((rows, cols))   
    
    for i in range(1,rows-1):
    	
    	for j in range(1,cols-1):
    		
    		#print(i,j)
    		#i,j es el elemento central
    		a = img[i,j] 
    		# i, j+1 derecha
    		b = img[i,j+1] 
    		# i, j-1 izquierda
    		c = img[i,j-1] 
    		# i-1, j, arriba
    		d = img[i-1,j] 
    		# i+1, j, Abajo
    		e = img[i+1,j] 
    		# i-1, j+1 diagonal derecha arriba
    		f = img[i-1,j+1] 
    		# i-1, j-1 diagonal izquierda arriba
    		g = img[i-1,j-1] 
    		# i+1, j+1 diagonal derecha abajo
    		h = img[i+1,j+1] 
    		# i+1, j-1 diagonal izquierda abajo 
    		o = img[i+1,j-1] 
    		window = np.array([a,b,c,d,e,f,g,h,o])
    		
    		new_matriz[i][j] = np.median(window)
    		
    		
 
    return new_matriz

#dilatacion
def expansion(img):
	img = otsu(img)
	#estructura en cruz
	B = np.matrix('0 1 0; 1 1 1; 0 1 0')

	rows, cols = img.shape
	new_matriz = np.zeros((rows, cols))

	for i in range(1,rows-1):

		for j in range(1,cols-1):
			

			if img[i,j] != 0:
				new_matriz[i,j] = img[i,j] 
				#print(i,j)
				
				# i, j+1 derecha
				if img[i,j+1] == 0:
					new_matriz[i,j+1] = 1
				# i, j-1 izquierda
				if img[i,j-1] == 0:
					new_matriz[i,j-1] = 1
				# i-1, j, arriba
				if img[i-1,j] == 0:
					new_matriz[i-1,j] = 1
				# i+1, j, Abajo
				if img[i+1,j] == 0:
					new_matriz[i+1,j] = 1
				# i-1, j+1 diagonal derecha arriba
				if img[i-1,j+1] == 0:
					new_matriz[i,j+1] = 1
				# i-1, j-1 diagonal izquierda arriba
				if img[i-1,j-1] == 0:
					new_matriz[i-1,j-1] = 1
				# i+1, j+1 diagonal derecha abajo
				if img[i+1,j+1] == 0:
					
					new_matriz[i+1,j+1] = 1

				# i+1, j-1 diagonal izquierda abajo 
				if img[i+1,j-1] == 0:
					new_matriz[i+1,j-1] = 1
		
		
		 
	return new_matriz

#erosion
def erosion(img):

	img = otsu(img)
	#estructura en cruz
	B = np.matrix('0 1 0; 1 1 1; 0 1 0')

	rows, cols = img.shape
	new_matriz = np.ones((rows, cols))

	for i in range(1,rows-1):

		for j in range(1,cols-1):

			flag = True
			if img[i,j] != 0:

				# i, j+1 derecha
				if img[i,j+1] == 0:
					flag = False
				# i, j-1 izquierda
				if img[i,j-1] == 0:
					flag = False
				# i-1, j, arriba
				if img[i-1,j] == 0:
					flag = False
				# i+1, j, Abajo
				if img[i+1,j] == 0:
					flag = False
				"""
				# i-1, j+1 diagonal derecha arriba
				if img[i-1,j+1] == 0:
					flag = False
				# i-1, j-1 diagonal izquierda arriba
				if img[i-1,j-1] == 0:
					flag = False
				# i+1, j+1 diagonal derecha abajo
				if img[i+1,j+1] == 0:
					flag = False
				# i+1, j-1 diagonal izquierda abajo
				if img[i+1,j-1] == 0:
					flag = False
				"""
			if flag:
				new_matriz[i,j] = 0

	return new_matriz

def kmeans(img, k):
	k = int(k[0])
	rows, cols = img.shape
	new_matriz = np.zeros((rows, cols))

	centroides = [3000,5000,7000]
	#centroides = calculate_centroides(img)

	groups = [[]]*k

	
	w = 0
	q = 0
	while True: 
		#no se puede hacer [[]]*k falla
		groups_tmp = [[],[],[]]
				
		for i in range(0,rows):

			for j in range(0,cols):
				
				minimo = abs(img[i][j] - centroides[0])
				pos = 0
				for x in range(1,len(centroides)):

					calc = abs(img[i][j] - centroides[x])
					
					if calc < minimo:
						pos = x
						minimo = calc

				groups_tmp[pos].append([i,j])


		if w == 0 :
			print ("w==0")
			groups = groups_tmp
			w = 1
			centroides = recalculate_centroides(img,groups_tmp)

		else:
			print ("w!=0")
			print ("recalculando centro",q)
			#verify_centroides(groups,groups_tmp)
			if np.array_equal(groups,groups_tmp):
				return calculate_img_kmean(img,groups)
			
			q+=1
			centroides = recalculate_centroides(img,groups_tmp)
			groups = groups_tmp


def calculate_centroides(img):
	rows, cols = img.shape
	cen1 = img[0][0]
	cen2 = img[math.floor(rows/2)][math.floor(cols/2)]
	cen3 = img[rows-1][cols-1]
	return [cen1,cen2,cen3]

def calculate_img_kmean(img,centroides):
	rows, cols = img.shape
	new_matriz = np.zeros((rows, cols))
	colores = [400,500,100]

	for i in range(0,len(centroides)):
		for valor in centroides[i]:
			new_matriz[valor[0]][valor[1]] = colores[i]

	return new_matriz



def recalculate_centroides(img,groups_tmp):
	new_centroides = []
	
	print(len(groups_tmp[0]),len(groups_tmp[1]),len(groups_tmp[2]))
	for group in groups_tmp:
		total = 0
		cantidad = 0
		for valor in group:
			total += img[valor[0]][valor[1]]
			cantidad += 1

		new_centroides.append(math.floor(total/cantidad))

	return new_centroides


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