import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation
import os
import skimage.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def Lenet1():
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    input_tensor = Input(shape=[28, 28, 1])
    # block1
    x = Conv2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    # block2
    x = Conv2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x)
    return model

def load_mnist_data(picture_path, name_path, num):
    label = np.zeros(10, dtype='float32')
    label[num] = 1
    name_arr = np.load(name_path)
    label_arr = np.empty((len(name_arr), 10), dtype='float32')
    img_arr = np.empty((len(name_arr), 28, 28, 1))
    for i in range(len(name_arr)):
        pic = picture_path + os.sep + name_arr[i]
        img = skimage.io.imread(pic)
        img = img[:, :, 0]
        print(img.shape)
        img = np.expand_dims(img, axis=0)
        img = img.transpose(1, 2, 0)
        img_arr[i] = img
        label_arr[i] = label
    img_arr = img_arr.astype('float32') / 255
    return img_arr, name_arr, label_arr

if __name__ == '__main__':
    model = Lenet1()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data("D:\PycharmProject\DeepBoundary\dataset\mnist\mnist.npz")
    train_datas = train_datas.reshape((train_datas.shape[0], 28, 28, 1)).astype('float32') #60000
    test_datas = test_datas.reshape((test_datas.shape[0], 28, 28, 1)).astype('float32') #10000
    train_datas /= 255
    test_datas /= 255
    train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)
    path = 'D:\PycharmProject\DeepBoundary\model\lenet1\lenet1.h5'
    checkpoint = ModelCheckpoint(filepath=path,
                                 monitor='val_accuracy', mode='auto', save_best_only='True')
    model.fit(train_datas, train_labels, epochs=20, batch_size=128, validation_data=(test_datas, test_labels), callbacks=[checkpoint])
    loss, acc = model.evaluate(test_datas, test_labels, batch_size=128)
    print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    print(model.summary())


    # loss, acc = model.evaluate(train_datas, train_labels, batch_size=128)
    # model是编译好的模型





