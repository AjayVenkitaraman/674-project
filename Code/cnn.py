import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras 
import sklearn
from sklearn.model_selection import KFold
import os
import glob
import numpy as np
import urllib

classes = ['cat','dog','bear','airplane',
                'ant','banana','bench','book',
                'bottlecap','bread']

url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
# Download the data of the aforementioned classes
for clas in classes:
	complete_url = url+clas+".npy"
	print("Downloading = ",complete_url)
	urllib.urlretrieve(complete_url, "./"+clas+".npy")

# Grep all the downloaded files and add them to a list
data_sets = glob.glob(os.path.join('./*.npy'))

#initialize variables 
input = np.empty([0, 784]) # Train data
labels = np.empty([0])	# Test data

index = 0
# Concat the train and test data from all the files
for file in data_sets:
	data = np.load(file)
	data = data[0: 6000, :]
	input = np.concatenate((input, data), axis=0)
	labels = np.append(labels, [index]*data.shape[0])
	index += 1

'''
	K-Folds cross-validator
	n_splits : Number of folds to be used
'''
n_fold = 5
kf = KFold(n_splits=n_fold,shuffle=True,random_state=9)
x_train = None
x_test = None
y_train = None
y_test = None
random_ordering = np.random.permutation(input.shape[0])
input = input[random_ordering, :]
labels = labels[random_ordering]
for train_index, test_index in kf.split(input):
    # Divide the dataset into train and test
    x_train, x_test = input[train_index], input[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    break

# Reshape the image size to be 28 x 28 
image_size = 28
x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1)
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1)

# Divide all the values by 255 to normalize the image
x_train /= 255.00
x_test /= 255.00
num_classes = len(classes)


# CNN Model
model = keras.Sequential()
model.add(layers.Convolution2D(64, (3, 3),
                        padding='same',
                        input_shape=x_train.shape[1:], activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))
model.add(layers.Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))
model.add(layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size =(3,3)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax')) 
optimizer = tf.train.AdamOptimizer()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
# Fit a model to the train data
model.fit(x = x_train, y = y_train, batch_size = 100,  validation_split = 0.2, epochs=15)

# Obtain the accuracy of the above model on the test data
accuracy = model.evaluate(x_test, y_test)
print('Test accuracy',accuracy[1] * 100)
