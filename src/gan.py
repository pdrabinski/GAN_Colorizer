import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

"""Generator"""
g_model = Sequential()
g_model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50000, 32, 32, 1)))
