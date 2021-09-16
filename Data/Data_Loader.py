import numpy as np

import tensorflow as tf

from tensorflow import keras


class Mnist_Data_Loader:
    def __init__(self, config):
        self.type_of_data = config['data_type']
        self.normal_class = config['normal_class']
        self.abnormal_class = config['abnormal_class']
        self.config = config

    def get_data(self):
        if self.type_of_data(self):
            self.mnist()

    def pretrain_mnist(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784)) / 255.0
        x_test = np.reshape(x_test, (-1, 784)) / 255.0

        x_train_normal = x_train[np.where(y_train == self.normal_class)]
        x_test_normal = x_test[np.where(y_test == self.normal_class)]
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_normal, x_train_normal))

        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.config['batch'])
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test_normal, x_test_normal))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(self.config['batch'])

        return train_dataset, test_dataset

    def svdd_mnist(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784)) / 255.0
        x_test = np.reshape(x_test, (-1, 784)) / 255.0

        x_train_normal = x_train[np.where(y_train == self.normal_class)]
        y_train_normal = y_train[np.where(y_train == self.normal_class)]

        x_test_normal_index = (y_test == self.normal_class)
        y_test_normal_index = (y_test == self.normal_class)
        x_test_abnormal_index = (y_test == self.abnormal_class)
        y_test_abnormal_index = (y_test == self.abnormal_class)

        y_test[y_test_normal_index] = 0 #normal_class
        y_test[y_test_abnormal_index] = 1 #abnormal_class

        x_test = x_test[x_test_normal_index | x_test_abnormal_index]
        y_test = y_test[y_test_normal_index | y_test_abnormal_index]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_normal, y_train_normal))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.config['batch'])

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(self.config['batch'])

        return train_dataset, test_dataset



