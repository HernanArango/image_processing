from tkinter import *
import pydicom
from kernel import Kernel
from thresholding import Thresholding
import cv2

class Interfaz:
	
	filename = 'MRI_Images/MRI01.dcm'

	def sobel(self):
		print ("ALGORTIMO SOBEL")
		dataset = pydicom.dcmread(self.filename)
		kernel = Kernel()
		Kernel.sobel(dataset.pixel_array)

	def gauss(self):
		print ("ALGORTIMO GAUSS")
		dataset = pydicom.dcmread(self.filename)
		Kernel.gauss(dataset.pixel_array)


	def histogram(self):
		print ("HISTOGRAMA")
		dataset = pydicom.dcmread(self.filename)
		Kernel.histogram(dataset.pixel_array)

	
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