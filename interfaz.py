from tkinter import filedialog
from tkinter import simpledialog
from tkinter import ttk
from tkinter import *


import pydicom
#import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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

		self.old_image = self.image
		self.show_img()
			
	def execute(self,function,color = 1,*options):

		#progress= ttk.Progressbar(self.root, orient = 'horizontal', maximum = 100, mode = 'indeterminate')
		#progress.start()
		#progress.pack(fill=BOTH)
		if self.typefile == "dcm":
			if options:
				self.image =function(self.image,options)
			else:
				self.image = function(self.image)
		else:
			if color == 0:
				#grayscale
				new_matriz = []
				img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
				#cv2.imwrite("test.png", img)
				if options:
					self.image = function(img,options)
				else:
					self.image = function(img)
				#return new_image
			else:
				new_matriz = []
				b,g,r = cv2.split (self.image)
				chanels_image = [b,g,r]
				for chanel in chanels_image:
					if options:
						new_matriz.append(function(chanel,options))
					else:
						new_matriz.append(function(chanel))

				self.image = cv2.merge([new_matriz[0],new_matriz[1],new_matriz[2]])
				cv2.imwrite("test.png", self.image)
				#return new_image
			
			
	def sobel(self):
		print ("ALGORITMO SOBEL")	
		self.execute(sobel,0)
		self.show(self.old_image,self.image)

	def strategy(self):
		print ("Estrategia")	
		#5 iteration
		self.execute(mediana)
		self.execute(mediana)
		self.execute(mediana)
		self.execute(mediana)
		self.execute(mediana)
		#kmeans
		self.execute(kmeans,0,5)

		self.show(self.old_image,self.image)


	def gauss(self):
		print ("ALGORITMO GAUSS")
		neighbours = simpledialog.askstring("", "Número de vecinos",
		parent=self.root)
		sigma = simpledialog.askstring("", "Sigma",
		parent=self.root)
		
		self.execute(gauss,0,neighbours,sigma)
		#self.new_image = cv2.GaussianBlur(self.image,(5,5),0)
		self.show(self.old_image,self.image)

	def raleight(self):
		print ("ALGORITMO RALEIGHT")
		neighbours = simpledialog.askstring("", "Número de vecinos",
		parent=self.root)
		sigma = simpledialog.askstring("", "Sigma",
		parent=self.root)
		self.execute(raleight,0,neighbours,sigma)
		self.show(self.old_image,self.image)

	def mediana(self):
		print ("ALGORITMO MEDIANA")
		self.execute(mediana,0)
		self.show(self.old_image,self.image)

	def expansion(self):
		print("Dilatacion")
		self.execute(expansion,0)
		self.show(self.old_image,self.image)

	def erosion(self):
		print("Erosion")
		self.execute(erosion,0)
		self.show(self.old_image,self.image)

	def kmeans(self):
		print("KMEANS")
		centroides = simpledialog.askstring("", "Número de centroides",
		parent=self.root)
		self.execute(kmeans,0,centroides)
		self.show(self.old_image,self.image)

	def otsu(self):
		print("Otsu")
		self.execute(otsu,0)
		self.show(self.old_image,self.image)

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
		if self.canvas != 0:
			#self.canvas.delete('all')
			self.canvas.get_tk_widget().destroy()
		fig = Figure(figsize=(6,6))
		plot=fig.add_subplot(111)
		plot.imshow(self.image,cmap=plt.cm.bone),plt.title('Original')
		self.canvas = FigureCanvasTkAgg(fig, master=self.root)
		self.canvas.get_tk_widget().pack()
		self.canvas.draw()
	
	def show_img2(self):
		plt.subplot(121),plt.imshow(self.old_image,cmap=plt.cm.bone),plt.title('Original')
		plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(self.image,cmap=plt.cm.bone),plt.title('Modificada')
		plt.xticks([]), plt.yticks([])
		plt.show()

		

	def show(self,img,new_img):
		if self.canvas != 0:
			self.canvas.get_tk_widget().destroy()
		fig = Figure(figsize=(6,6))
		plt1=fig.add_subplot(121)
		plt2=fig.add_subplot(122)
		
		plt1.imshow(img,cmap=plt.cm.bone),plt.title('Original')
		plt.xticks([]), plt.yticks([])
		plt2.imshow(new_img,cmap=plt.cm.bone),plt.title('Convolucion')
		plt.xticks([]), plt.yticks([])
		#plt.show()

		self.canvas = FigureCanvasTkAgg(fig, master=self.root)
		self.canvas.get_tk_widget().pack()
		self.canvas.draw()
		"""
		plt.subplot(121),plt.imshow(img,cmap=plt.cm.bone),plt.title('Original')
		plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(new_img,cmap=plt.cm.bone),plt.title('Convolucion')
		plt.xticks([]), plt.yticks([])
		plt.show()
	
		"""
	def change_color(self,option):
		print("color y gray",self.color.get(),self.grayscale.get())

		if option == 0:
			self.color.set(False)
			self.grayscale.set(True)
		else:
			self.color.set(True)
			self.grayscale.set(False)



	def __init__(self):
		self.root = Tk()
		self.root.geometry("500x500")
		self.root.title("Proyecto Imagenes")
		 
		menubar = Menu(self.root)
		self.root.config(menu=menubar)
		 
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="Abrir", command=self.open)
		filemenu.add_separator()
		filemenu.add_command(label="Salir", command=self.root.quit)


		self.color = BooleanVar()
		self.color.set(True)
		self.grayscale= BooleanVar()
		self.grayscale.set(False)


		
		

		view_menu = Menu(menubar)
		view_menu.add_command(label="Ver imagen",command=self.show_img2)
		view_menu.add_command(label="Histograma", command=self.histogram)
		view_menu.add_command(label="Estrategia propuesta", command=self.strategy)
		view_menu.add_separator()
		view_menu.add_checkbutton(label="Escala de grises", onvalue=1, offvalue=False, command= lambda :self.change_color(0),variable = self.grayscale)
		view_menu.add_checkbutton(label="Color", onvalue=True, offvalue=0, command=lambda: self.change_color(1),variable = self.color)
		

		editmenu = Menu(menubar, tearoff=0)
			
		
		

		morfologicas = Menu(editmenu, tearoff=0)
		morfologicas.add_command(label="Dilatacion", command=self.expansion)
		morfologicas.add_command(label="Erosion", command=self.erosion)
		editmenu.add_cascade(label="Operaciones Morfologicas", menu=morfologicas)

		filtros = Menu(editmenu, tearoff=0)
		filtros.add_command(label="Raleight",command=self.raleight)
		filtros.add_command(label="Mediana", command=self.mediana)
		filtros.add_command(label="Gauss",command=self.gauss)
		editmenu.add_cascade(label="Filtros", menu=filtros)

		umbral = Menu(editmenu, tearoff=0)
		umbral.add_command(label="Otsu", command=self.otsu)
		editmenu.add_cascade(label="Umbral", menu=umbral)

		agrupacion = Menu(editmenu, tearoff=0)
		agrupacion.add_command(label="Kmeans", command=self.kmeans)
		editmenu.add_cascade(label="Agrupación", menu=agrupacion)
		
		bordes = Menu(editmenu, tearoff=0)
		bordes.add_command(label="Sobel", command=self.sobel)
		editmenu.add_cascade(label="Bordes", menu=bordes)

		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="Ayuda")
		helpmenu.add_separator()
		helpmenu.add_command(label="Acerca de ...")
		 
		menubar.add_cascade(label="Archivo", menu=filemenu)
		menubar.add_cascade(label="Algoritmos", menu=editmenu)
		menubar.add_cascade(label='Opciones', menu=view_menu)
		menubar.add_cascade(label="Ayuda", menu=helpmenu)

		self.canvas = 0
		self.root.mainloop()


Interfaz()