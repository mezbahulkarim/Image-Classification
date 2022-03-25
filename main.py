# %%
#CREDIT FOR THE CODE: https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/16_cnn_cifar10_small_image_classification/cnn_cifar10_dataset.ipynb

from tensorflow.keras import datasets, layers, models

# %%
#DATASET 
data = datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = data
y_train=y_train.reshape(-1, )        

#NORMALIZING DATA
x_train = x_train/255
x_test = x_test/255

# %%
#SET UP CNN
cnn = models.Sequential([
    
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),        
    layers.MaxPooling2D((2,2)),     

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
      
    layers.Flatten(),           
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
    ])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

# %%
#TRAIN
cnn.fit(x_train, y_train, epochs=15)             


# %%
#TEST
cnn.evaluate(x_test, y_test)         
