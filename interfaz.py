from tkinter import *
import pydicom
import matplotlib.pyplot as plt
from tools.kernel import *
import cv2

class Interfaz:
	
	filename = 'MRI_Images/MRI01.dcm'
	

	def sobel(self):
		print ("ALGORITMO SOBEL")
		dataset = pydicom.dcmread(self.filename)
		new_image = sobel(dataset.pixel_array)
		self.show(dataset.pixel_array,new_image)


	def gauss(self):
		print ("ALGORITMO GAUSS")
		dataset = pydicom.dcmread(self.filename)
		new_image = gauss(dataset.pixel_array)
		self.show(dataset.pixel_array,new_image)


	def histogram(self):
		print ("HISTOGRAMA")
		dataset = pydicom.dcmread(self.filename)
		hist = histogram(dataset.pixel_array)
		histX = hist[0]
		histY = hist[1]
		plt.xlabel('Values')
		plt.ylabel('Frecuency')
		plt.title('Histogram')
		plt.plot(histX,histY, 'k')
		plt.grid(True)
		plt.show()

	def show_img(self):
		dataset = pydicom.dcmread(self.filename)
		img = grayscale(dataset.pixel_array)
		plt.imshow(img,cmap=plt.cm.bone),plt.title('Original')
		plt.show()

	def show(self,img,new_img):
		
		plt.subplot(121),plt.imshow(img,cmap=plt.cm.bone),plt.title('Original')
		plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(new_img,cmap=plt.cm.bone),plt.title('Convolucion')
		plt.xticks([]), plt.yticks([])
		plt.show()
	
	def __init__(self):
		root = Tk()
		 
		menubar = Menu(root)
		root.config(menu=menubar)
		 
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="Abrir")
		filemenu.add_separator()
		filemenu.add_command(label="Salir", command=root.quit)


		show_all = BooleanVar()
		show_all.set(True)
		show_done = BooleanVar()
		show_not_done = BooleanVar()

		view_menu = Menu(menubar)
		view_menu.add_checkbutton(label="eliminar bordes", onvalue=1, offvalue=False, variable=show_all)
		view_menu.add_checkbutton(label="otro", onvalue=True, offvalue=0, variable=show_done)
		#view_menu.add_checkbutton(label="Show Not Done", onvalue=1, offvalue=0, variable=show_not_done)



		editmenu = Menu(menubar, tearoff=0)
		editmenu.add_command(label="Ver imagen",command=self.show_img)
		editmenu.add_command(label="Gauss",command=self.gauss)
		editmenu.add_command(label="Sobel", command=self.sobel)
		editmenu.add_command(label="Histograma", command=self.histogram)
		

		 
		 
		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="Ayuda")
		helpmenu.add_separator()
		helpmenu.add_command(label="Acerca de ...")
		 
		menubar.add_cascade(label="Archivo", menu=filemenu)
		menubar.add_cascade(label="Algoritmos", menu=editmenu)
		menubar.add_cascade(label='Opciones', menu=view_menu)
		menubar.add_cascade(label="Ayuda", menu=helpmenu)


		root.mainloop()


Interfaz()