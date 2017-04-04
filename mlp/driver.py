import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import mlp

#ask the user for number of nodes in each hidden layer
num_layers = int(raw_input("enter the number of hidden layers :  "))

num_neurons = []
num_neurons.append(784)
for i in range(1,num_layers+1):
    num_neurons.append(int(raw_input('enter the hidden nodes in '+ str(i) + 'th layer :  ')))
num_neurons.append(10)

activ_func = []
for i in range(1,num_layers+1):
    activ_func.append(raw_input('enter the activation function of your choice tanh ReLU '+ str(i) + 'layer :  '))

print num_neurons
print activ_func

net = mlp.Network(num_neurons,activ_func,'softmax')
net.SGD(training_data, 5, 10, 0.05,'rmsprop', test_data=validation_data)

