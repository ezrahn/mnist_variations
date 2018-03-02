import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD

import mnist

train_images = mnist.train_images()[:5000]
train_labels = keras.utils.to_categorical(mnist.train_labels()[:5000], num_classes=10)
test_images = mnist.test_images()[:500]
test_labels = keras.utils.to_categorical(mnist.test_labels()[:500], num_classes=10)

model = Sequential()
model.add(Dense(196, activation='sigmoid', input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(98, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20, batch_size=128)
score = model.evaluate(test_images, test_labels, batch_size=128)
                            