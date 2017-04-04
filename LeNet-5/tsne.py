#Once you train your data use this snippet of code in jupyter notebook 
#to obtain the tsne plots

#Use the network weights to extract features to visualize them using t-SNE
weights = net.weights
#Features from the last conv layer
def feature_extractor_cnn(x):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        max1_del = np.empty((6,28,28))
        max2_del = np.empty((16,10,10))
        
        conv1_out = conv2d(x, weights[0]) #First convolution layer
        conv1_out_act = tanh(conv1_out)
        max1_out,max1_del = maxpool3d(conv1_out_act) #First maxpool layer
        conv2_out = conv3d(max1_out, weights[1]) #Second convolution layer
        conv2_out_act = tanh(conv2_out)
        max2_out, max2_del = maxpool3d(conv2_out_act)
        
        #We have a 16 * ( 5*5) tensor after all these operations
        #flatten it and use it as input to the mlp
        
        mlp_inp = flat(max2_out)
        return mlp_inp

#Features from the last FC layer
def feature_extractor_fc(x):
        conv1_out = conv2d(x,weights[0]) #First convolution layer
        conv1_out = tanh(conv1_out)
        max1_out = maxpool3d_feedfwd(conv1_out) #First maxpool layer
        conv2_out = conv3d(max1_out,weights[1]) #Second convolution layer
        conv2_out = tanh(conv2_out)
        max2_out = maxpool3d_feedfwd(conv2_out)
        
        #We have a 16 * ( 5*5) tensor after all these operations
        #flatten it and use it as input to the mlp
        
        mlp_inp = flat(max2_out)
        mlp_inp = np.array([mlp_inp])
        mlp = mlp_inp.T
        
        fc1 = mlp
        fc2 = np.dot(weights[2], fc1)
        fc2_act = tanh(fc2)
        fc3 = np.dot(weights[3], fc2_act)
        fc3_act = tanh(fc3)

        return flat(fc3_act)




from matplotlib import pyplot as plt
from tsne import bh_sne

tsne_data = processed_validation_data[:1000]

#t-SNE using cnn output as feature vectors
final_tsne_data = []
for data in tsne_data:
    final_tsne_data.append(feature_extractor_cnn(data[0]))


final_tsne_data = np.asarray(final_tsne_data).astype('float64')

final_tsne_data_label = []
for data in tsne_data:
    final_tsne_data_label.append(data[1])

vis_data = bh_sne(final_tsne_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=final_tsne_data_label, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

#t-SNE using fc layer output as feature vectors
final_tsne_data = []
for data in tsne_data:
    final_tsne_data.append(feature_extractor_fc(data[0]))


final_tsne_data = np.asarray(final_tsne_data).astype('float64')

final_tsne_data_label = []
for data in tsne_data:
    final_tsne_data_label.append(data[1])

vis_data = bh_sne(final_tsne_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=final_tsne_data_label, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
