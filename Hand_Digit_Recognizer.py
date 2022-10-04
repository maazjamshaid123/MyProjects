from PIL import Image
import mnist
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#training
x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()

x_train = x_train.reshape((-1,28*28))
x_test = x_test.reshape((-1,28*28))

x_train = (x_train/256)
x_test = (x_test/256)

model = MLPClassifier(solver = 'adam', activation='relu', hidden_layer_sizes=(64,64))
model.fit(x_train, y_train)

prediction = model.predict(x_test)
acc = confusion_matrix(y_test, prediction)

#Testing
img = Image.open('an.png')

data = list(img.getdata())

for i in range(len(data)):
    data[i] = 255 - data[i]

data = np.array(data)/256
number = model.predict([data])

print(number)
