import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD

import mnist

use_full_set = True
train_size = 5000
test_size = 500
epochs = 150
batch_size = 128

if use_full_set:
       train_images = mnist.train_images()
       train_labels = keras.utils.to_categorical(mnist.train_labels(), num_classes=10)
       test_images = mnist.test_images()
       test_labels = keras.utils.to_categorical(mnist.test_labels(), num_classes=10)
else:
       train_images = mnist.train_images()[:train_size]
       train_labels = keras.utils.to_categorical(mnist.train_labels()[:train_size], num_classes=10)
       test_images = mnist.test_images()[:test_size]
       test_labels = keras.utils.to_categorical(mnist.test_labels()[:test_size], num_classes=10)

model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
score = model.evaluate(test_images, test_labels, batch_size=batch_size)
print score