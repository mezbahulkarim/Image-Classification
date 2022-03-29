#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers


#%%
#LOAD DATA
train_data = pd.read_csv("D:/Py/CNN/dataset/fashion-mnist_train.csv", sep=',')
test_data = pd.read_csv("D:/Py/CNN/dataset/fashion-mnist_test.csv", sep=',')

# %%
#VISUALIZE DATA
train_data.head()
test_data.head()

# %%
#Convert to Numpy Array
train_data = np.array(train_data, dtype = 'float32')
test_data = np.array(test_data, dtype='float32')


# %%
#TRAIN TEST SPLIT
x_train = train_data[:,1:]/255      #all rows without first column 
y_train = train_data[:,0]
x_test= test_data[:,1:]/255
y_test=test_data[:,0]

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)  #??????

#%%
#SHOW IMAGES 
class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.Figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape((28,28)))
    label_index = int(y_train[i])
    plt.title(class_names[label_index])
plt.show()

# %%
#MORE VISUALIZATION
W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (16,16))
axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array
n_train = len(train_data) # get the length of the train dataset

# Select a random number from 0 to n_train
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_train)
    # read and display an image with the selected index    
    axes[i].imshow( train_data[index,1:].reshape((28,28)) )
    labelindex = int(train_data[index,0])
    axes[i].set_title(class_names[labelindex], fontsize = 9)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.3)


# %%
#Image Shape
image_rows = 28
image_cols = 28
batch_size = 4096
image_shape = (image_rows,image_cols,1) 

x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)

# %%
#THE MODEL
cnn_model = models.Sequential([
    layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
    layers.MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
    layers.Dropout(0.2),
    layers.Flatten(), # flatten out the layers
    layers.Dense(32,activation='relu'),
    layers.Dense(10,activation = 'softmax')
    
])

cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam',metrics =['accuracy'])

#%%
#TRAINING
history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=4096,
    epochs=5,
    verbose=1,
    validation_data=(x_validate,y_validate),
)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training - Loss Function')

plt.subplot(2, 2, 2)
plt.plot(history.history['acc'], label='Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.title('Train - Accuracy')


# %%
score = cnn_model.evaluate(x_test,y_test,verbose=0)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Accuracy : {:.4f}'.format(score[1]))

# %%
