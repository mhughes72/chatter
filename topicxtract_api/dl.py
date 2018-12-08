from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.models import Sequential

from util import constants


def model84_short( ):
    model = Sequential()
    model.add(Dense(1024, input_shape=(880,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(constants.SHORT_LABELS, activation='softmax'))
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-04)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def model84_long( ):
    model = Sequential()
    model.add(Dense(1024, input_shape=(1000,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(constants.LONG_LABELS, activation='softmax'))
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-04)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
