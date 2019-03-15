import numpy as np
from six.moves import cPickle as pickle
from six.moves import range


from dense import DenseNetwork

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    #Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

model = DenseNetwork()
model.input((28, 28))
model.add(196, 'sigmoid')
model.add(58, 'sigmoid')
model.add(10, 'softmax') # output layer

x_train, y_train = reformat(x_train, y_train)
x_test, y_test = reformat(x_test, y_test)

x_axis, y_axis = model.fit(x_train, y_train, 1, learning_rate=0.1)

prediction = model.predict(x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, prediction)
print(score)

import matplotlib.pyplot as plt
plt.scatter(x_axis, y_axis)
plt.show()