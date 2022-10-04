import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test, y_test)= datasets.cifar10.load_data()
x_train = x_train/255
x_test = x_test/255

cnn = models.Sequential([
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
cnn.fit(x_train, y_train, epochs=10)

cnn.evaluate(x_test, y_test)
y_test = y_test.reshape(-1,)
classes = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']

def plot_sample(x,y,index):
  plt.figure(figsize = (15,2))
  plt.imshow(x[index])
  plt.xlabel(classes[y[index]])
  
y_pred = cnn.predict(x_test)
y_pred[:5]

y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]
y_test[:5]

n = 7001
plot_sample(x_test, y_test,n)
classes[y_classes[n]]
