import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an color image in grayscale
img = cv2.imread('imagen.png',0)



def convolucion(imagen,matriz_convolucion):
    
   
    rows, cols = imagen.shape

    matriz_convolucionada = np.zeros((rows, cols))
    
    print(imagen.shape)
    
    for i in range(1,rows-1):
    	
    	for j in range(1,cols-1):
    		
    		#print(i,j)
    		#i,j es el elemento central
    		a = imagen[i,j] * matriz_convolucion[1,1]
    		# i, j+1 derecha
    		b = imagen[i,j+1] * matriz_convolucion[1,2]
    		# i, j-1 izquierda
    		c = imagen[i,j-1] * matriz_convolucion[1,0]
    		# i-1, j, arriba
    		d = imagen[i-1,j] * matriz_convolucion[0,1]
    		# i+1, j, Abajo
    		e = imagen[i+1,j] * matriz_convolucion[2,1]
    		# i-1, j+1 diagonal derecha arriba
    		f = imagen[i-1,j+1] * matriz_convolucion[0,2]
    		# i-1, j-1 diagonal izquierda arriba
    		g = imagen[i-1,j-1] * matriz_convolucion[0,0]
    		# i+1, j+1 diagonal derecha abajo
    		h = imagen[i+1,j+1] * matriz_convolucion[2,2]
    		# i+1, j-1 diagonal izquierda abajo 
    		o = imagen[i+1,j-1] * matriz_convolucion[2,0]
    		matriz_convolucionada[i,j] = a+b+c+d+e+f+g+h+i
    		print (i,j)
    		print(matriz_convolucionada[i,j])
    		print (a+b+c+d+e+f+g+h+o)
    		
    	
    	"""
    	print(matriz_concolucionada[i,j])
    	print (a+b+c+d+e+f+g+h+i)
    	break
    	"""
    print ("convolucionando en forma")
    return matriz_convolucionada

matriz_convolucion = np.matrix("1, 2, 1; 2, 4, 2; 1, 2, 1")

#print (np.size(img,1))
#print (np.size(img,0))


new_img = convolucion(img,matriz_convolucion)
print ("a mostrar")
plt.imshow(new_img, cmap=plt.cm.bone)
plt.show()
    