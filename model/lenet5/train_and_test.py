import tensorflow as tf
from keras import backend as k
import keras
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Reshape
import os
from keras.models import load_model


def Lenet5():
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    input_tensor = Input(shape=[28, 28, 1])
    # block1
    x = Conv2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    # block2
    x = Conv2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x)
    return model

if __name__ == '__main__':
    model = Lenet5()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data("D:\PycharmProject\DeepBoundary\dataset\mnist\mnist.npz")
    train_datas = train_datas.reshape((train_datas.shape[0], 28, 28, 1)).astype('float32') #60000
    test_datas = test_datas.reshape((test_datas.shape[0], 28, 28, 1)).astype('float32') #10000
    train_datas /= 255
    test_datas /= 255
    train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)
    path = 'D:\PycharmProject\DeepBoundary\model\lenet5\lenet5.h5'
    model.fit(train_datas, train_labels, epochs=10, batch_size=128, verbose=1)

    loss, acc = model.evaluate(test_datas, test_labels, batch_size=128)
    print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    print('')
    print("train over")

    loss, acc = model.evaluate(train_datas, train_labels, batch_size=128)
    print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    print('')
    print("train over")

    model.save(path)



