# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()

model.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
model.add(Activation('relu'))

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


