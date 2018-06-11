from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import tkFileDialog
from Tkinter import *
import tkMessageBox
from glob import glob
#hay algunos import que no uso porque tengo la costumbre de dejarlos xDXD
######################
#este codigo es una red neuronal con deep learning, para aprender su funcionamiento primero hay que definir que es deep learning.
#El Deep Learning lleva a cabo el proceso de Machine Learning usando una red neuronal artificial que se compone de un numero de niveles jerarquicos.
#En el nivel inicial de la jerarquia la red aprende algo simple y luego envia esta informacion al siguiente nivel.
#El siguiente nivel toma esta informacion sencilla, la combina, compone una informacion algo un poco mas compleja, y se lo pasa al tercer nivel, y asi sucesivamente.
#el ejemplo mas basico el del gato, el nivel inicial de una red de Deep Learning podria utilizar las diferencias entre las zonas claras y oscuras de una imagen
#para saber donde estan los bordes de la imagen. El nivel inicial pasa esta informacion al segundo nivel, que combina los bordes construyendo formas simples,
#como una linea diagonal o un angulo recto. El tercer nivel combina las formas simples y obtiene objetos mas complejos como ovalos o rectangulos.
#El siguiente nivel podria combinar los ovalos y rectangulos, formando barbas, patas o colas rudimentarias.
#El proceso continua hasta que se alcanza el nivel superior en la jerarquia, en el cual la red aprende a identificar gatos.
#El deep learning ha obtenido mucha atencion debido a que principalmente obtiene tasas de exito elevadas con entrenamiento 'no supervisado'.
#En el caso del ejemplo, las redes de Deep Learning aprenderian a identificar gatos aunque las imagenes no tuvieran la etiqueta 'gato'.
#Por esa razon se decidio a usar deep learning ya que no ocupa tags.
#Para un mejor aprovechamiento de la red, es recomendable menos capas y mas neuronas, se puede hacer pruebas hasta encontrar el mejor resultado,
#a prueba y error.
####################
#########codigo bien chidori
######aqui se carga el csv resultado del load2 osea el out.csv
#PD: no es necesario usar el segundo combine, ya que puede entrenar directamente desde el primer boton.
def load1():
    global fname
    global v
    fname = tkFileDialog.askopenfilename(filetypes = (("Csv files", "*.csv"), ("All files", "*")))
    print fname
    v.set(fname)
    return fname
#######primero se usa esta funcion aqui se carga el csv
def load2():
    global fname1
    global v1
    fname1 = tkFileDialog.askopenfilename(filetypes = (("Csv files", "*.csv"), ("All files", "*")))
    print fname1
    v1.set(fname1)

    ##en esta se concatena el nombre del fname1 por ejemplo si se llama x.csv se le agregara el x1.csv y la salida sera out.csv
    fout=open("out.csv","a")
    # first file:
    for line in open(fname1):
        fout.write(line)

    # now the rest:
    for num in range(1,2):
        f = open(fname1)
        f.next() # skip the header
        for line in f:
             fout.write(line)
        f.close() # not really needed
    fout.close()
##############

##########

#CSV file
#de del load1 se manda a esta funcion que lo que hace en palabras simples lee el csv cargado y lo ingresa en una lista llamada dataset.
def load_csv(fname):
	dataset = list()
	with open(fname, 'r') as file:
		csv_reader = reader(file,delimiter=',')
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
######################


#string column to float
#aqui se convierten las columnas a flotante.
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
##################
############################
# Convert string column to integer
#en esta funcion es similar a la anterior pero se convierten de str a enteros.
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
#################
# aqui encuentra la funcion min y maxima de cada columna.
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
##############
#Se realiza un Rescale de las columnas del dataset para el rango 0 a 1.
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
##################
#se evalua el algoritmo usango K-fold cross validation con N folds, eso significa que es 201/n=n O, n records para cada fold.
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
##########################
#para la precision
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
#
#Se evalua el algoritmo usando la cross validation split.
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
#
#se activan las neuronas para una entrada.
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

#
#Se puede transferir la activacion de la funcion usando la funcion sigmoid.
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

#
#Se implementa el forward propagation, para el row de los datos
#de nuestro dataset con la red neuronal.
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

#
#Se utiliza la funcion de transferencia sigmoid.
def transfer_derivative(output):
	return output * (1.0 - output)

#
#se registra el error y se almacena en las nueronas.
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

#
#se actualizan los pesos para la red,
#ingresando el row de los datos,
#un learning rate que asume que el forward y el backproagation ya se realizaron.
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

#
#la red es actualizada usando la stochastic gradient descent,
#esto involucra el primer ciclo para los epoch,
#y por cada epoch se actualiza la red para cada row en el dataset de training,
#Porque las actualizaciones son echas para cada patron de entrenamiento,
#este tipo de aprendizaje se llama online learning,
#Si los errores se acumulan atraves de cada epoch, antes de la actualizacion de los pesos,
#esto es llamado batch learning, o batch gradient descent.
#En esta funcion se implementa el entrenamiento de una red ya inicializada,
#con su respectivo dataset, learning rate, el numero de epoch, y los posibles valores de salida,
#Los valores esperados de las salida, son usados para la transferencia de clase, en el training data,
# adentro de un one hot encoding que significa "One hot encoding es un proceso por el cual las variables son convertidas en una forma de datos
# que provee para los algoritmos de Machine Learning, hacer un mejor trabajo en la prediccion"
#en palabras simples es:
#Un vector binario para cada columa de cada valor de clase para qe haga match,
#la salida de la red requiere calcular el error del output layer.

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

#
#Se crea la red neuronal para el entrenamiento, que acepta tres parametros,
#el numero de entradas, el numero de neuronas en el hidden layer,
# y el numero de salidas.
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

#
#Regresa el index de la salida de la red, que tiene la probabilidad mas alta,
#esto asume que el valor de la clase tiene que ser convertido tiene entero empezando del 0.
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

#
#Esta funcion fue realizada para administrar la aplicacion de el algoritmo de backpropagation,
#primero se inicializa la red, se entrena en base a el dataset de entrenamiento y
#luego se usa la red entrenada para realiar la predicion del test dataset.
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)


#########################
#aqui es donde ocurre la magia, se ingresan los valores y se asigna el dataset,
#se normaliza el dataset, y se hace uso de las funciones.
#tambien aqui se imprimen los resultados.en el textarea.
def load():
	n1=float(caja1.get())
	n2=int(caja2.get())
	n3=int(caja3.get())
    ######################
        n4=int(caja4.get())
    ############
	#
	#
	#
	dataset = load_csv(fname)

	for i in range(len(dataset[0])-1):
		str_column_to_float(dataset, i)
	#
	str_column_to_int(dataset, len(dataset[0])-1)
	#
	minmax = dataset_minmax(dataset)
	normalize_dataset(dataset, minmax)
	#
	#############

	n_folds = n4
	l_rate = n1
	n_epoch = n3
	n_hidden = n2
	scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
	txt.insert(INSERT, 'Scores %s\n' % scores)
	txt.insert(END,'Mean Accuracy: %.3f%%\n' % (sum(scores)/float(len(scores))) )

#################################
#las instrucciones para saber como funciona.
def about():
    tkMessageBox.showinfo("Mensaje","Primero usar combine luego usar load, no necesariamente tambien se puede entrenar directamente.")
###############3
#para limpiar el text area.
def clear():
	txt.delete('1.0', END)
#############
#########screen
#de aqui para abajo se crea toda la interfaz grafica.
gui=Tk()
gui.title("Clasificador")
gui.geometry("600x550+600+550")
################
#creo que estas variables no sirven pero igual no afectan en nada xdxd
v=StringVar()
v1=StringVar()
#################### se definen los label y las cajas
var1=StringVar()
var1.set("Learning rate : ")
label1 = Label(gui,textvariable=var1,height = 2)
label1.pack()
numero1=StringVar()
caja1=Entry(gui,bd=4, textvariable=numero1)
caja1.pack()
#####
var2=StringVar()
var2.set("Capas intermedias : ")
label2=Label(gui,textvariable=var2,height=2)
label2.pack()
numero2=StringVar()
caja2=Entry(gui,bd=4, textvariable=numero2)
caja2.pack()
#####
var3=StringVar()
var3.set("Epoch  : ")
label3=Label(gui,textvariable=var3,height=2)
label3.pack()
numero3=StringVar()
caja3=Entry(gui,bd=4, textvariable=numero3)
caja3.pack()
#########################
var4=StringVar()
var4.set("Neuronas : ")
label4=Label(gui,textvariable=var4,height=2)
label4.pack()
numero4=StringVar()
caja4=Entry(gui,bd=4, textvariable=numero4)
caja4.pack()
####################
#para los btones se crea un frame.
frame = Frame(gui)
frame.pack()
button = Button(frame,text="Train",command=load)
button.pack(side=LEFT)
button2 = Button(frame,text="Clear",command=clear)
button2.pack(side=LEFT)
###############el text area junto con la scrollbar.
txt_frm =Frame(gui, width=100, height=200)
txt_frm.pack(fill="both", expand=False)
txt_frm.grid_propagate(False)
####
txt_frm.grid_rowconfigure(0, weight=1)
txt_frm.grid_columnconfigure(0, weight=1)
txt =Text(txt_frm, borderwidth=3, relief="sunken")
txt.config(font=("arial", 12), undo=True, wrap='word')
txt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
###########lo mismo que arriba.
scrollb =Scrollbar(txt_frm, command=txt.yview)
scrollb.grid(row=0, column=1, sticky='nsew')
txt['yscrollcommand'] = scrollb.set
##########
##############
#el menu bien chidori.
menubar = Menu(gui)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_separator()
filemenu.add_command(label="Load", command=load1)
filemenu.add_command(label="Combine", command=load2)
filemenu.add_command(label="Exit", command=gui.quit)
helpmenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=filemenu)
helpmenu.add_command(label="About...", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)
###########3
gui.config(menu=menubar)
##################
#######################3
#el mainloop:V
gui.mainloop()
