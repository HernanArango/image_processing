from tkinter import filedialog
from tkinter import *
import pydicom
import matplotlib.pyplot as plt
from tools.kernel import *
import cv2 

class Interfaz:
	
	filename = 'MRI_Images/MRI01.dcm'
	
	def open(self):
		
		self.filename = filedialog.askopenfilename(initialdir = "",title = "Select file",filetypes = (("dicom files","*.dcm"),("Jpg","*.jpg"),("Png","*.png"),("all files","*.*")))
		print(self.filename)
		self.typefile = (self.filename.split('/')[-1]).split(".")[-1]
		
		if self.typefile == "dcm":
			dataset = pydicom.dcmread(self.filename)
			self.image = dataset.pixel_array
		else:
			self.image = cv2.imread(self.filename)
			
	def execute(self,function,color = 1):
		
		if self.typefile == "dcm":
			return function(self.image)
		else:
			if color == 0:
				#grayscale
				new_matriz = []
				img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
				#cv2.imwrite("test.png", img)
				new_image = function(img)
				return new_image
			else:
				new_matriz = []
				b,g,r = cv2.split (self.image)
				chanels_image = [b,g,r]
				for chanel in chanels_image:
					new_matriz.append(function(chanel))

				new_image = cv2.merge([new_matriz[0],new_matriz[1],new_matriz[2]])
				cv2.imwrite("test.png", new_image)
				return new_image
			
			
	def sobel(self):
		print ("ALGORITMO SOBEL")	
		new_image = self.execute(sobel)
		self.show(self.image,new_image)


	def gauss(self):
		print ("ALGORITMO GAUSS")
		new_image = self.execute(gauss,0)
		#new_image = cv2.GaussianBlur(self.image,(5,5),0)
		self.show(self.image,new_image)

	def raleight(self):
		print ("ALGORITMO RALEIGHT")
		new_image = self.execute(raleight,0)
		self.show(self.image,new_image)

	def mediana(self):
		print ("ALGORITMO MEDIANA")
		new_image = self.execute(mediana,0)
		self.show(self.image,new_image)

	def expansion(self):
		print("Dilatacion")
		new_image = self.execute(expansion,0)
		self.show(self.execute(otsu),new_image)

	def erosion(self):
		print("Erosion")
		new_image = self.execute(erosion,0)
		self.show(self.execute(otsu),new_image)

	def kmeans(self):
		print("KMEANS")
		new_image = self.execute(kmeans,0)
		self.show(self.image,new_image)

	def otsu(self):
		print("Otsu")
		new_image = self.execute(otsu,0)
		self.show(self.image,new_image)

	def histogram(self):
		print ("HISTOGRAMA")
		hist = histogram(self.image)
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
		img =dataset.pixel_array
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
		filemenu.add_command(label="Abrir", command=self.open)
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
		editmenu.add_command(label="Raleight",command=self.raleight)
		editmenu.add_command(label="Mediana", command=self.mediana)
		editmenu.add_command(label="Sobel", command=self.sobel)
		editmenu.add_command(label="Dilatacion", command=self.expansion)
		editmenu.add_command(label="Erosion", command=self.erosion)
		editmenu.add_command(label="Kmeans", command=self.kmeans)
		editmenu.add_command(label="Histograma", command=self.histogram)
		editmenu.add_command(label="Otsu", command=self.otsu)
		

		 
		 
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