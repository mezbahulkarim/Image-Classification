# %%
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np



# %%
data = datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = data

#x_test.shape                        #images, dimension, dimension, rgb channels
#y_train[:5]
y_train=y_train.reshape(-1, )        #convert 2d array to 1d
y_train[:5]

#plt.figure(figsize=(15,2))     
#plt.imshow(x_train[50])             #change index see image

# %%
   






