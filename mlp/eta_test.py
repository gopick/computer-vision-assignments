#before you run this code comment out the plot section in network.py
#this code let's you choose the best eta for a particular activation function and a particular adaptive learning algorithm
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network

etas =[5.0,3.0,1.0,0.5,0.1,0.06,0.01,0.005]
for eta in etas:
	net = network.Network([784, 30, 10],'tanh','softmax')
	net.SGD(training_data, 5, 10, eta,'adagrad', test_data=validation_data)


for eta in etas:
	net = network.Network([784, 30, 10],'tanh','softmax')
	net.SGD(training_data, 5, 10, eta,'rmsprop', test_data=validation_data)

for eta in etas:
	net = network.Network([784, 30, 10],'ReLU','softmax')
	net.SGD(training_data, 5, 10, eta,'adagrad', test_data=validation_data)


for eta in etas:
	net = network.Network([784, 30, 10],'ReLU','softmax')
	net.SGD(training_data, 5, 10, eta,'rmsprop', test_data=validation_data)