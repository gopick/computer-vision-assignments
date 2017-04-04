import random
# Third-party libraries
import numpy as np
import csv
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes,activation,cost_fn):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.cost_fn = cost_fn

        ######################################################
        #If activation funtion of different layers is different
        self.activation = []
        self.activation_deriv = []
        for fn in activation:
            if fn == 'tanh':
                self.activation.append(tanh)
                self.activation_deriv.append(tanh_deriv)
            elif fn == 'ReLU':
                self.activation.append(ReLU)
                self.activation_deriv.append(ReLU_derivative)
        #The derivative of softmax is explicitly calculated and used in function backprop hence needn't 
        if cost_fn == 'softmax':
            self.activation.append(softmax)
        elif cost_fn == 'quadLoss':
            self.activation.append(tanh)
            self.activation_deriv.append(tanh_deriv)    

        with open('data.csv', 'ab+') as csvfile:
            titlewriter = csv.writer(csvfile, delimiter=',')
            titlewriter.writerow(activation)        

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        i=0
        for b, w in zip(self.biases, self.weights):
            a = self.activation[i](np.dot(w, a)+b)
            i = i+1
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, adaptive_learn,
            test_data= None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""
        plot_accur=[]
        plot_epoch=[]
        ada_b = [np.zeros(b.shape) for b in self.biases]
        ada_w = [np.zeros(w.shape) for w in self.weights]
        rms_b = [np.zeros(b.shape) for b in self.biases]
        rms_w = [np.zeros(w.shape) for w in self.weights]

        with open('data.csv', 'ab+') as csvfile:
            titlewriter = csv.writer(csvfile, delimiter=',')
            titlewriter.writerow(['epochs', 'eta', 'adaptive learning algorithm'])
            titlewriter.writerow([epochs, eta, adaptive_learn])
(
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                if adaptive_learn == 'adagrad':
                    ada_w,ada_b =self.update_mini_batch(mini_batch, eta, 'adagrad',ada_w,ada_b)
                elif adaptive_learn == 'rmsprop':
                    rms_w,rms_b =self.update_mini_batch(mini_batch, eta, 'rmsprop',rms_w,rms_b)
            if test_data:
                plot_epoch.append(j+1)
                evaluation = self.evaluate(test_data)
                plot_accur.append(100-(evaluation/100))
                print "Epoch {0}: {1} / {2}".format(
                    j, evaluation , n_test)
                with open('data.csv', 'ab+') as csvfile:
                    titlewriter = csv.writer(csvfile, delimiter=',')
                    titlewriter.writerow([j,evaluation,10000])

                    
            else:
                print "Epoch {0} complete".format(j)
        #code for plotting error vs epoch 
        plt.plot(plot_epoch, plot_accur,'ro-')
        plt.xlabel('epochs')
        plt.ylabel('err% ')
        plt.axis([0, 55, 0, 20])
        plt.show()

    def update_mini_batch(self, mini_batch, eta, adaptive_learn,ar_w,ar_b):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        epi = 1e-8
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: #x is input y is output of training data
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        if adaptive_learn == 'adagrad':
            ar_w = [nw*nw+aw for nw,aw in zip(nabla_w,ar_w)]
            ar_b = [nb*nb+ab for nb,ab in zip(nabla_b,ar_b)]
        elif adaptive_learn == 'rmsprop':
            ar_w = [(0.1*nw*nw)+ 0.9*aw for nw,aw in zip(nabla_w,ar_w)]
            ar_b = [(0.1*nb*nb)+ 0.9*ab for nb,ab in zip(nabla_b,ar_b)]

        if adaptive_learn == 'rmsprop':
            self.weights = [w-(eta/len(mini_batch))*np.divide(nw,np.sqrt(aw+epi))
                            for w, nw,aw in zip(self.weights, nabla_w ,ar_w)]
            self.biases = [b-(eta/len(mini_batch))*np.divide(nb,np.sqrt(ab+epi))
                        for b, nb,ab in zip(self.biases, nabla_b,ar_b)]

        elif adaptive_learn == 'adagrad':
            self.weights = [w-(eta/len(mini_batch))*np.divide(nw,np.sqrt(aw+epi))
                            for w, nw,aw in zip(self.weights, nabla_w ,ar_w)]
            self.biases = [b-(eta/len(mini_batch))*np.divide(nb,np.sqrt(ab+epi))
                        for b, nb,ab in zip(self.biases, nabla_b,ar_b)]
        return(ar_w,ar_b)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        i=0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation[i](z)
            activations.append(activation)
            i=i+1

        if self.cost_fn == 'softmax':
            delta = (activations[-1] - y))
        elif self.cost_fn == 'quadLoss':
            delta = self.cost_derivative(activations[-1], y) * self.activation_deriv(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_deriv[-l+1](z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        #To find test error or validation error, use this code
        return sum(int(x == y) for (x, y) in test_results)
        ##################################################
        # To find train error use this code
        # return sum(int(x == list(y).index(max(list(y)))) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#Here we define softmax which we will apply for the last layer

def softmax(x):
    soft_max = np.exp(x -np.max(x))
    return (soft_max /soft_max.sum())
    
# Here we define the tanh function.
def tanh(x):
    return np.tanh(x)

# Here we define the tanh function derivative.
def tanh_deriv(x):
    return (1.0 - (np.tanh(x)**2))


# Here we define the ReLU function.
def ReLU(x):
    """ReLU returns 1 if x>0, else 0."""
    return np.maximum(0,x)

# Here we define the ReLU function derivative.
def ReLU_derivative(x):
    return 1. * (x > 0)
