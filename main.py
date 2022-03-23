# %%
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# %%
#DATASET WORK
data = datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = data

#x_test.shape                        #images, dimension, dimension, rgb channels
#y_train[:5]

y_train=y_train.reshape(-1, )        #convert 2d array to 1d
#y_train[:5]

#plt.figure(figsize=(15,2))     
#plt.imshow(x_train[50])             #change index, see images

#NORMALIZING DATA
x_train = x_train/255
x_test = x_test/255

# %%
#SET UP CNN
cnn = models.Sequential([
    #cnn part
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),            #detects features(32 boxes)  kernel_sizes(sub boxes to detect featuers)
    layers.MaxPooling2D((2,2)),      #compresses feature extraction

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    #dense part
    layers.Flatten(),                 #1D now
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
    ])

#%%

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

# %%
#TRAIN
cnn.fit(x_train, y_train, epochs=15)             


# %%
#TEST
cnn.evaluate(x_test, y_test)        #~70% ACCURACY   