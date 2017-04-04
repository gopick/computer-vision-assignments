import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

plot_accur = [92.52,97.77,98.40,98.72,98.92]
plot_itr = [1,2,3,4,5]
plt.plot(np.asarray(plot_itr),np.asarray(plot_accur), '-o')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy using KERAS (%) ')
plt.show()
