import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an color image in grayscale
#img = cv2.imread('lena.jpg')

"""
def convolucion(imagen,matriz_convolucion):
    
    #rows = imagen.shape
    i=0
    matriz_convolucionada = []
    print (matriz_convolucionada)
    
    for canales_imagen in imagen:
        matriz_convolucionada.append(aux_convolucion(canales_imagen,matriz_convolucion))
        i=i+1
    #print(len(matriz_convolucionada),i)
    #print(matriz_convolucionada)
    #return matriz_convolucionada[0] + matriz_convolucionada[1] + matriz_convolucionada[2]
    return cv2.merge((matriz_convolucionada[0],matriz_convolucionada[1],matriz_convolucionada[2]))
"""
def aux_convolucion(imagen,matriz_convolucion):
    
   
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
    		matriz_convolucionada[i,j] = a+b+c+d+e+f+g+h+o
    		#print (i,j)
    		#print(matriz_convolucionada[i,j])
    		#print (a+b+c+d+e+f+g+h+o)
    		
 
    print ("convolucionando en forma")
    return matriz_convolucionada


def split_channels(img):
    #new_img = convolucion(img,matriz_convolucion)
    #b,g,r = cv2.split(img)

    #Blue
    b = img.copy()
    b[:,:,1]=0
    b[:,:,2]=0
    #Green
    g = img.copy()
    g[:,:,0]=0
    g[:,:,2]=0
    #Red
    r = img.copy()
    r[:,:,0]=0
    r[:,:,1]=0

    return r,g,b

def save_image(new_img):
    cv2.imwrite( "nueva.png", new_img );
    plt.imshow(new_img,cmap=plt.cm.bone)
    plt.show()

    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#matriz_convolucion = np.matrix("1, 2, 1; 2, 4, 2; 1, 2, 1")
#matriz_convolucion = np.matrix("0, 1, 0; 1, -4, 1; 0, 1, 0")


#r,g,b = split_channels(img)

#print (np.size(img,1))
#print (np.size(img,0))


def convolution(img,matrix_convolution):
    #matriz_convolucion = np.ones((3,3),np.float32)/20
    #matriz_convolucion = np.matrix("0, 1, 0; 1, -4, 1; 0, 1, 0")
    #b,g,r = cv2.split (img)
    #new_img = convolucion([b,g,r],matriz_convolucion)
    new_img = aux_convolucion(img,matrix_convolution)

    # plot the image using matplotlib
    """
    plt.subplot(121),plt.imshow(img,cmap=plt.cm.bone),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(new_img,cmap=plt.cm.bone),plt.title('Convolucion')
    plt.xticks([]), plt.yticks([])
    plt.show()
    """
    return new_img
    #save_image(new_img)



