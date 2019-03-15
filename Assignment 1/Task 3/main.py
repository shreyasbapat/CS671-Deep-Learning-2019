import tensorflow as tf
import keras
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range


from dense import DenseNetwork

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train / 255.0
#x_test = x_test / 255.0

image_size = 28
num_labels = 10

def reformat(dataset, labels):
       dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
       #Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
       labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
       return dataset, labels
       
model = DenseNetwork()
model.input((784, 1))
model.add(16, 'sigmoid')
model.add(16, 'sigmoid')
model.add(10, 'softmax') # output layer

x_train, y_train = reformat(x_train, y_train)
x_test, y_test = reformat(x_test, y_test)

model.fit(x_train, y_train, 10, learning_rate=0.02)

prediction = model.predict(x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, prediction)
print(score)