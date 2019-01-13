import matplotlib.pyplot as plt
import numpy as np
import math
import keras

# Importar la API keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.optimizers import Adam
import keras.callbacks as cb

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
	self.losses.append(batch_loss)
    

#Cargo los datos
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

#Se definen las variables
img_size = 28
img_size_flat = img_size * img_size
# Tupla con la altura y el ancho de las imagenes utilizadas para remodelar matrices.
# Esto se utiliza para pintar las imagenes.
img_shape = (img_size, img_size)
# Tupla con altura, anchura y profundidad utilizada para remodelar matrices.
# Esto se usa para remodelar en Keras.
img_shape_full = (img_size, img_size, 1)
# Numero de canales de color para las imagenes: 1 canal para escala de grises.
num_channels = 1
# Numero de clases, una clase para cada uno de 10 digitos.
num_classes = 10


#Construccion de la red neuronal de forma secuencial
model = Sequential()
# La entrada es una matriz aplanada con 784 elementos (img_size * img_size),
# pero las capas convolucionales esperan imagenes con forma (28, 28, 1), por tanto hacemos un reshape
model.add(Reshape(img_shape_full))
# Primera capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
# Segunda capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
# Aplanar la salida de 4 niveles de las capas convolucionales
# a 2-rank que se puede ingresar a una capa totalmente conectada 
model.add(Flatten())
# Primera capa completamente conectada  con ReLU-activation.
model.add(Dense(128, activation='relu'))
# Ultima capa totalmente conectada con activacion de softmax para usar en la clasificacion.
model.add(Dense(num_classes, activation='softmax'))


#Anadir funcion de coste, un optimizador y las metricas de rendimiento
optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


#Fase de entrenamiento
print("Comienza el entrenamiento")
history = LossHistory()
model.fit(x=data.train.images, y=data.train.labels, callbacks=[history], epochs=15, batch_size=128)
result = model.evaluate(x=data.train.images, y=data.train.labels)
for name, value in zip(model.metrics_names, result):
    print(name, value)

#Pruebo con el conjunto de test
print("Pruebo el conjunto de test")
result = model.evaluate(x=data.test.images, y=data.test.labels)
for name, value in zip(model.metrics_names, result):
    print(name, value)

print("")
model.summary()


plt.switch_backend('agg')
plt.ioff()
fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(history.losses)
ax.set_title('Batch losses')

plt.show()
fig.savefig('img1.png')


