import numpy as np

import tensorflow as tf

from tensorflow import keras


class Data_Loader:
    def __init__(self, args):
        self.type_of_data = args['data_type']
        self.one_class_data = args['normal_class']
        self.args = args

    def get_data(self):
        if self.type_of_data(self):
            self.mnist()

    def pretrain_mnist(self):
        batch_size = self.args['batch']
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784)) / 255.0
        x_test = np.reshape(x_test, (-1, 784)) / 255.0

        x_train_one_class = x_train[np.where(y_train == self.one_class_data)]
        y_train_one_class = y_train[np.where(y_train == self.one_class_data)]

        x_test_one_class = x_test[np.where(y_test == self.one_class_data)]


        normal_index = (y_test == self.one_class_data)
        abnormal_index = (y_test != self.one_class_data)

        y_test[normal_index] = 0 #normal class
        y_test[abnormal_index] = 1 # abnormal class

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_one_class, x_train_one_class))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test_one_class, x_test_one_class))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

        return train_dataset, test_dataset

    def svdd_mnist(self):
        batch_size = self.args['batch']
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784)) / 255.0
        x_test = np.reshape(x_test, (-1, 784)) / 255.0

        x_train_one_class = x_train[np.where(y_train == self.one_class_data)]
        y_train_one_class = y_train[np.where(y_train == self.one_class_data)]

        normal_index = (y_test == self.one_class_data)
        abnormal_index = (y_test != self.one_class_data)

        y_test[normal_index] = 0 #normal class
        y_test[abnormal_index] = 1 # abnormal class

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_one_class, y_train_one_class))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

        return train_dataset, test_dataset



