#!/bin/python2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal
from get_data import get_data_gen

model = Sequential()

model.add(Conv2D(32, 5, strides=5, input_shape=(50, 50, 1),
                 activation='relu', padding='same',
                 kernel_initializer=glorot_normal()))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, 3, strides=3, activation='relu', padding='same',
                 kernel_initializer=glorot_normal()))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer=glorot_normal()))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_normal()))

early_stop = EarlyStopping(monitor='loss', min_delta=0.025, patience=5)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

test_split = 0.3
train_data, train_labels, test_data, test_labels, generator =\
    get_data_gen(test_split=test_split)
model.fit_generator(generator.flow(train_data, train_labels, batch_size=32),
                    steps_per_epoch=100, epochs=100, callbacks=[early_stop])

print(model.evaluate(test_data, test_labels, batch_size=200))
