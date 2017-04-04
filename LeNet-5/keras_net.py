import keras
import os
os.environ['KERAS_BACKEND'] ='tensorflow'

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

from keras.layers import Dense, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D,Convolution3D,MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

#To build the LeNet5 network
K.set_image_dim_ordering('tf') #Since we are using tensorflow as the backend

batch_size = 128
nb_classes = 10
nb_epoch = 5

#input image dims
img_rows,img_cols = 28,28

#pool size
pool_size = (2,2)
#the number output of filters in the convolution
conv1_filters = 6
conv2_filters = 16
#convolution 2D kernel size 
kernel_size_conv1 = (5,5)
kernel_size_conv2 = (5,5)

#Load the mnist data and reshape it into 28*28*1 images
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows,img_cols,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1).astype('float32')
X_train = X_train/255
X_test = X_test/255

#one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

cnn_model = Sequential()
cnn_model.add(Convolution2D(conv1_filters, kernel_size_conv1[0], kernel_size_conv1[1],border_mode = 'valid',input_shape=(img_rows,img_cols,1)))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=pool_size))

cnn_model.add(Convolution2D(conv2_filters ,kernel_size_conv2[0],kernel_size_conv2[1],border_mode = 'valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=pool_size))

cnn_model.add(Flatten())
cnn_model.add(Dense(120))
cnn_model.add(Activation('tanh'))
cnn_model.add(Dense(84))
cnn_model.add(Activation('tanh'))
cnn_model.add(Dense(num_classes))
cnn_model.add(Activation('softmax'))

cnn_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
cnn_model.fit(X_train, y_train,validation_data=(X_test, y_test), nb_epoch=5,batch_size=batch_size)              

train_loss = cnn_model.history['loss']
val_loss   = cnn_model.history['val_loss']
train_acc  = cnn_model.history['acc']
val_acc    = cnn_model.history['val_acc']
xc = range(100)

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.style.use(['classic'])
plt.show()