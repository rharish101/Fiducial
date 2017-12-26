#!/usr/bin/env python
from __future__ import print_function
from builtins import input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal, Constant
from get_data import get_data

model = Sequential()

model.add(Conv2D(32, 5, strides=5, input_shape=(50, 50, 1),
                 activation='relu', padding='same',
                 kernel_initializer=glorot_normal(),
                 bias_initializer=Constant(0.1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, 3, strides=3, activation='relu', padding='same',
                 kernel_initializer=glorot_normal(),
                 bias_initializer=Constant(0.1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer=glorot_normal(),
                bias_initializer=Constant(0.1)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_normal(),
                bias_initializer=Constant(0.1)))

early_stop = EarlyStopping(monitor='loss', min_delta=0.025, patience=5)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

test_split = 0.3
train_data, train_labels, test_data, test_labels = get_data()
#model.fit_generator(generator.flow(train_data, train_labels, batch_size=32),
                    #steps_per_epoch=100, epochs=100, callbacks=[early_stop])
model.fit(train_data, train_labels, batch_size=32, epochs=100,
          callbacks=[early_stop])
print(model.evaluate(test_data, test_labels, batch_size=200))

response = input("Do you want to save this model? (Y/n): ")
if response.lower() not in ['n', 'no', 'nah', 'nein', 'nahi', 'nope']:
    model.save('template_0_detect.h5')
    print("Model saved")

