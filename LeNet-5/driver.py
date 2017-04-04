import mnist_loader
import LeNet_final
import numpy as np

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

sizes = [[6, 5, 5], [16, 6, 5, 5], 400, 120, 84, 10]

net = LeNet_final.Network(sizes)
#def SGD(self,processed_training_data, epochs, mini_batch_size, eta,test_data=None)
#Validation accuracy 16 min batch
#net.SGD(processed_training_data[:6400], 2, 16, 0.1, test_data= processed_validation_data[:300])        
#training error min batch 64
#net.SGD(processed_training_data[:6400], 2, 64, 0.1, test_data=processed_training_data[:200])        
#training error min batch 128
#net.SGD(processed_training_data[:6400], 2, 128, 0.1, test_data=processed_training_data[:200])        
#training accuracy min batch 16
net.SGD(processed_training_data[:1024], 2, 128, 0.1, test_data=processed_training_data[:200])        

print 'forward conv:' ,np.sum(np.array(net.fwd))/len(net.fwd) ,'forward fc',np.sum(np.array(net.bkd))/len(net.bkd) ,'backward fc',np.sum(np.array(net.back_bkd))/len(net.back_bkd),'backward conv',np.sum(np.array(net.back_fwd))/len(net.back_fwd)