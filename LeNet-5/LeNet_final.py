import csv
import random
#Construction of the LeNet5 network
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import numpy as np
from scipy import signal
import time

class Network(object):
    
    def __init__(self, sizes):
        #Initializing the weights and biases for all the layers in the LeNet5
        #You have a choice to choose the kernel size and number of kernels
        #The depth is constant and follows LeNet5
        self.num_layers = len(sizes)
        self.weights =[]
        conv1 = np.random.randn(sizes[0][0],sizes[0][1],sizes[0][2])*5/(1*np.sqrt(sizes[0][0]*sizes[0][1]*sizes[0][2]))
        self.weights.append(conv1)
        conv2 =[]
        for i in range (sizes[1][0]):
            conv2.append(np.random.randn(sizes[1][1],sizes[1][2],sizes[1][3])*5/(1*np.sqrt(sizes[1][1]*sizes[1][2]*sizes[1][3])))
        conv2 = np.array(conv2)
        self.weights.append(conv2)
        wts = [np.random.randn(y, x)*5/(1*np.sqrt(x*y)) for x, y in zip(sizes[2:][:-1], sizes[2:][1:])]
        #wts = [np.random.randn(400, 120)*5/(1*np.sqrt(400*120)), np.random.randn(120, 84)*5/(1*np.sqrt(120*84)), np.random.randn(84, 10)*5/(1*np.sqrt(84*10))]
        for wt in wts:
            self.weights.append(wt)

        self.sizes = sizes
        self.fwd = []
        self.bkd = []
        self.back_fwd = []
        self.back_bkd = []
#################################################################
    #Feedforward step of the LeNet5
    #Return the prediction given an input image of size 32*32 
    def feedforward(self,a):
        #conv1_out = np.empty((self.sizes[0][0],a.shape[0]-self.sizes[0][1],a.shape[1]-self.sizes[0][2]) #Use this if you are using different convolutions
                                                    #than specified in the LeNet5 paper
        
        #This is for standard LeNet5
        #Using this just to reduce the number of flops
        
        conv1_out = conv2d(a,self.weights[0]) #First convolution layer
        conv1_out = tanh(conv1_out)
        max1_out = maxpool3d_feedfwd(conv1_out) #First maxpool layer
        conv2_out = conv3d(max1_out, self.weights[1]) #Second convolution layer
        conv2_out = tanh(conv2_out)
        max2_out = maxpool3d_feedfwd(conv2_out)
        
        #We have a 16 * ( 5*5) tensor after all these operations
        #flatten it and use it as input to the mlp
        
        mlp_inp = flat(max2_out)
        mlp_inp = np.array([mlp_inp])
        mlp = mlp_inp.T
        
        fc1 = mlp
        fc2 = np.dot(self.weights[2], fc1)
        fc2_act = tanh(fc2)
        fc3 = np.dot(self.weights[3], fc2_act)
        fc3_act = tanh(fc3)
        fc4 = np.dot(self.weights[4], fc3_act)
        y_pred = softmax(fc4) 
        
        return y_pred
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        plot_accur=[]
        plot_itr=[]
        rms_w = [np.zeros(w.shape) for w in self.weights]
        if test_data: n_test = len(test_data)
        n = len(training_data)
        counter_min_batch = 0
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta,rms_w)
                if test_data:
                    evaluation = self.evaluate(test_data)
                    #plot_accur.append(100-(evaluation/100))
                    plot_accur.append(evaluation/2)
                    plot_itr.append(counter_min_batch)
                    print "Epoch {0}: {1} / {2}".format(
                        j, evaluation , n_test)
                    with open('data_train_64.csv', 'ab+') as csvfile:
                        titlewriter = csv.writer(csvfile, delimiter=',')
                        titlewriter.writerow([counter_min_batch,evaluation,100])
                else:
                    print "Epoch {0} complete".format(j)
                counter_min_batch = counter_min_batch + 1
    
        #plot the accuracies
        a = plt.plot(np.asarray(plot_itr),np.asarray(plot_accur), label = 'Accuracy')
        plt.xlabel('Number of iterations')
        plt.ylabel('Accuracy for two hundred training examples ')
        plt.show()
    
    def update_mini_batch(self, mini_batch, eta,ar_w):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        epi = 1e-8
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #Backprop is called and the weight is changed for all the kernels
            
            delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        ar_w = [(0.1*nw*nw)+ 0.9*aw for nw,aw in zip(nabla_w,ar_w)]
        self.weights = [w-(eta/len(mini_batch))*np.divide(nw,np.sqrt(aw+epi))
                            for w, nw,aw in zip(self.weights, nabla_w ,ar_w)]

      
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        conv_int=time.time()
        del_w = [np.zeros(w.shape) for w in self.weights]
        
        max1_del = np.empty((6,28,28))
        max2_del = np.empty((16,10,10))
        
        conv1_out = conv2d(x,self.weights[0]) #First convolution layer
        conv1_out_act = tanh(conv1_out)
        max1_out,max1_del = maxpool3d(conv1_out_act) #First maxpool layer
        conv2_out = conv3d(max1_out, self.weights[1]) #Second convolution layer
        conv2_out_act = tanh(conv2_out)
        max2_out, max2_del = maxpool3d(conv2_out_act)
        
        #We have a 16 * ( 5*5) tensor after all these operations
        #flatten it and use it as input to the mlp
        conv_final = time.time()
        self.fwd.append(conv_final - conv_int)
        mlp_inp = flat(max2_out)
        mlp_inp = np.array([mlp_inp])
        mlp = mlp_inp.T
        #Backprop for mlp
        
        fc1 = mlp
        fc2 = np.dot(self.weights[2], fc1)
        fc2_act = tanh(fc2)
        fc3 = np.dot(self.weights[3], fc2_act)
        fc3_act = tanh(fc3)
        fc4 = np.dot(self.weights[4], fc3_act)
        y_pred = softmax(fc4)      
        feed_final = time.time()
        self.bkd.append(feed_final - conv_final)
        

        back_feed_int = time.time()
        delta = (y_pred - y) * softmax_prime(fc4) #Cross entropy loss function with softmax
        
        del_w[-1] = np.dot(delta, fc3_act.transpose())
        
        delta = np.dot(self.weights[-1].transpose(), delta) * tanh_prime(fc3)
        del_w[-2] = np.dot(delta, fc2_act.transpose())
        
        delta = np.dot(self.weights[-2].transpose(), delta) * tanh_prime(fc2)
        del_w[-3] = np.dot(delta, fc1.transpose())
        
        delta = np.dot(self.weights[-3].transpose(), delta) #delta of 400. The result of maxpooling
        
        #this delta is (400,1)
        #gotta reshape this for maxpool
        delta =delta.reshape((16,5,5))
        
        temp_delta = []
        for kernel in delta:
            kernel = np.repeat(kernel,2,axis=1)
            kernel = np.repeat(kernel,2,axis=0)
            temp_delta.append(kernel)
        temp_delta = np.array(temp_delta)
        
        #Now we use the filter max2_del to send back the gradient
        #only through the maximum of the 2-D matrices
        delta = max2_del * temp_delta #(16,10,10)
        delta =  delta * tanh_prime(conv2_out) 
        #deltas.insert(0,delta)
        back_feed_final = time.time()
        self.back_bkd.append(back_feed_final - back_feed_int)
        del_w[-4] = conv3d_backprop_tensor_wt(max1_out,delta)
        
        delta = conv3d_backprop_tensor_del(delta,self.weights[1]) #(6,14,14)
        
        temp_del = []
        for kernel in delta:
            kernel = np.repeat(kernel,2,axis=1)
            kernel = np.repeat(kernel,2,axis=0)
            temp_del.append(kernel)
        temp_del = np.array(temp_del)
        
        #Now we use the filter max1_del to send back the gradient
        #only through the maximum of the 2-D matrices
        delta = max1_del * temp_del #(6,28,28)
        delta = delta * tanh_prime(conv1_out)
        
        
        del_w[-5] = conv2d_backprop_tensor_wt(x,delta)
        back_conv_final = time.time()
        self.back_fwd.append(back_conv_final-back_feed_final)
        return del_w
             
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        #TO TEST ON TRAINING DATA
        test_results = [(np.argmax(self.feedforward(x)),np.argmax(y))
                        for (x, y) in test_data]
        #TO TEST ON VALIDATION DATA
        # test_results = [(np.argmax(self.feedforward(x)),y)
        #                 for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def trainloss(self,x,y):
        return cost_fn(softmax(self.feedforward(x)),y)
        



#ALL THE FUNCTIONS USED ARE DEFINED HERE

#flatten a tensor into a vector
def flat(inp_tensor):
    return inp_tensor.flatten()


#takes the spatial dimensions of the tensor required and gives us the tensor from a vector
def maketensor(inp_vec,x,y,z):
    return inp_vec.reshape((z,y,x))


#Give it a tensor, gives back tensor after maxpooling
def maxpool3d_feedfwd(inp_tensor):
    N,h,w = inp_tensor.shape
    return inp_tensor.reshape(N, h / 2, 2, w / 2, 2).max(axis=(2, 4))


#Give it a tensor, gives back tensor after maxpooling
#Also gives you a filter storing the indices where the maximum existed
def maxpool3d(inp_tensor):
    out_shape = inp_tensor.shape[1]/2
    pooled_out=np.empty((inp_tensor.shape[0],out_shape,out_shape))
    backprop_tensor = np.empty((inp_tensor.shape))
    i=0
    for inp in inp_tensor:
        pooled_out[i],backprop_tensor[i] = maxpool2d(inp)
        i=i+1
    return pooled_out,backprop_tensor

#Given a 2-d array it does max-pool and reduces the input into half 
def maxpool2d(inp):
    rows_dims = inp.shape[0]/2
    cols_dims = inp.shape[1]/2
    backprop_filter = np.zeros((inp.shape[0], inp.shape[1]))
    maxpooled = np.zeros((rows_dims, cols_dims))
    
    maxpooled_row = 0
    for rows in range(0,inp.shape[0]-1,2):
        row_offset = rows +1
        maxpooled_col = 0
        for cols in range(0,inp.shape[1]-1,2):
            col_offset = cols +1
            
            maxpooled[maxpooled_row][maxpooled_col] = inp[np.ix_([rows,row_offset],[cols,col_offset])].max()
            max_index = inp[np.ix_([rows,row_offset],[cols,col_offset])].argmax()
            backprop_filter[rows + max_index/2][cols + max_index%2] = 1
            maxpooled_col = maxpooled_col + 1
        
        maxpooled_row = maxpooled_row + 1
        
    return maxpooled,backprop_filter

def ReLU(x):
    return x*(x > 0)

def ReLU_prime(x):
    return 1*(x > 0)

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1.0 - (np.tanh(z))**2.0    

def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def softmax_prime(x):
    a =  softmax(x) 
    return a*(1-a)

#Given an input 3-D delta input and 4-D tensor of weights
#out puts the delta of the precious layer
#returns a 3d tensor of delta
#Right now the implementation doesn't use 
#np.repeat to make a tensor of delta of the previous layer
def conv3d_backprop_tensor_del(delt,wts):
    out = []
    for i,j in zip(delt,wts):
        rot_i = np.rot90(i,2)#10*10
        temp =[]
        for k in j:#k is 5*5
            #temp.append(signal.convolve(padded_i,np.rot90(k,2),'valid'))
            temp.append(signal.correlate(k,rot_i))#Rotated by 180  #size 14*14
        temp = np.array(temp) #14*14*6
        out.append(temp)
    out = np.array(out)#16 of 14*14*6 so (16,6,14,14)
    return out.sum(axis=0) #back to (6,14,14)           
    

#Given an input 2-D previous layer input and 3-D tensor of gradients
#out puts the gradient for weights
#returns a 3d tensor of delta of weights
def conv2d_backprop_tensor_wt(inp,conv):
    out=[]
    for i in conv:
        out.append(signal.correlate(inp,i,mode ='valid'))
    return np.array(out)


#Given an input 3-D previous layer input and 3-D tensor of gradients
#out puts the gradient for weights
#returns a 4d tensor of delta of weights
def conv3d_backprop_tensor_wt(inp,conv):
    out=[]
    for i in conv:
        temp = []
        for j in inp:
            temp.append(signal.correlate(j,i,mode ='valid'))
        temp = np.array(temp)
        out.append(temp)
    return np.array(out)

#Given an input 3-D tensor and 4-D convloution list
#4-D because it is a list of 3-D convolution kernels
#Returns a convoluted 3-D tensor. Third dimension is the number of kernels
def conv3d(inp,conv):
    out =[]
    for i in conv:#5*5*6 16 times
        temp = []
        for j,k in zip(inp,i):#14*14 and 5*5
            temp.append(signal.correlate(j,k,mode='valid'))#10*10
        temp = np.array(temp)#(6,10,10)
        out.append(np.sum(temp,axis=0))#10*10 appends 16 times
    return np.array(out) #returns (16,10,10)

#Given an input 2-D and 3-D convolution list
#3-D because it is a list of 2-D convolution kernels
#Returns a convoluted 3-D tensor. Third dimension is the number of kernels
def conv2d(inp,conv):
    out = []
    for i in conv:
        out.append(signal.correlate(inp,i,mode ='valid'))
    return np.array(out)