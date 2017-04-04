import numpy as np
from scipy import signal,ndimage
import scipy.io as scio
import sys
import time
from matplotlib import pyplot as plt
import matplotlib as mpl
import mnist_loader
import LeNet_final

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

new_training_data = [trainingdata[0].reshape((28,28)) for trainingdata in training_data]
padded_training_data = [np.lib.pad(traindata, 2, padwithzeros) for traindata in new_training_data]
processed_training_data = [[padded_training, train_data[1]]for  padded_training,train_data in zip(padded_training_data,training_data)]

new_validation_data = [validdata[0].reshape((28,28)) for validdata in validation_data]
padded_validation_data = [np.lib.pad(valdata, 2, padwithzeros) for valdata in new_validation_data]

#You should also process validation data so that it can be used a 32*32 input
processed_validation_data = [[padded_valid, valid_data[1]]for  padded_valid,valid_data in zip(padded_validation_data,validation_data)]

#Data is ready

#Network defined
sizes = [[6, 5, 5], [16, 6, 5, 5], 400, 120, 84, 10]

net = LeNet_final.Network(sizes)

def cost_fn(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)))

def softmax(x):
    soft_max = np.exp(x -np.max(x))
    return (soft_max /soft_max.sum())

trainLabel = processed_validation_data[0][1]
trainData = processed_validation_data[0][0]

dW = net.backprop(trainData,trainLabel)
dW0 = dW[3].flatten() #This decides which layer we are plotting for

# For plotting
num = 500
j = 3
x=[] 
for i in range(num):
    x.append(i)
y= []
z= []


epsi = 1e-04
W0 = net.weights[j].flatten()
temp = W0
for i in range(num):

    base = temp[i]
    base += epsi
    W0[i] = base
    net.weights[j] = W0.reshape(net.weights[j].shape)    
    one = cost_fn(softmax(net.feedforward(trainData)),trainLabel)
    
    base = temp[i]
    base -= epsi
    W0[i] = base
    net.weights[j] = W0.reshape(net.weights[j].shape) 
    two = cost_fn(softmax(net.feedforward(trainData)),trainLabel)

    W0[i] = temp[i]
    
    val1 = dW0[i]
    val2 = (one-two)/(2.0*epsi)
    y.append(val1)
    z.append(val2)  

#PLOT

a = plt.plot(x,(np.asarray(y)-np.asarray(z)), label = 'Difference')
plt.legend()
plt.show()

c = plt.plot(x,np.asarray(y), label = 'Actual')
plt.legend()
plt.show()

b = plt.plot(x,np.asarray(z) , label = 'Ideal')
plt.legend()
plt.show()
