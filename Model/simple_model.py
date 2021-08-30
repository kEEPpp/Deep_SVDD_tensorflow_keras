import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd

from icecream import ic
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import layers, losses
# from tensorflow.keras import losses
from tensorflow.keras.models import Model


class Pretrain_AutoEncoder(Model):
    def __init__(self, hidden1, hidden2, latent, input_dim):
        super(Pretrain_AutoEncoder, self).__init__()
        self.input_dim = input_dim

        self.encoder_layer1 = layers.Dense(hidden1, activation='relu', name='encoder1')
        self.encoder_layer2 = layers.Dense(hidden2, activation='relu', name='encoder2')

        self.latent = layers.Dense(latent, activation='relu', name='latent')

        self.decoder_layer1 = layers.Dense(hidden2, activation='relu', name='decoder1')
        self.decoder_layer2 = layers.Dense(hidden1, activation='relu', name='decoder2')
        self.outputs = layers.Dense(self.input_dim, activation='relu', name='outputs')

    def build_graph(self):
        inputs_ = layers.Input(shape=self.input_dim, name='inputs')
        return Model(inputs=inputs_, outputs=self.call(inputs_))

    def call(self, input_data, **kwargs):
        x = self.encoder_layer1(input_data)
        x = self.encoder_layer2(x)
        x = self.latent(x)
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = self.outputs(x)

        return x


class DeepSVDD(Model):
    def __init__(self, hidden1, hidden2, latent, input_dim):
        super(DeepSVDD, self).__init__()
        self.encoder_layer1 = layers.Dense(hidden1, activation='relu', name='encoder1')
        self.encoder_layer2 = layers.Dense(hidden2, activation='relu', name='encoder2')

        self.latent = layers.Dense(latent, activation='relu', name='latent')

        self.input_dim = input_dim

    def build_graph(self):
        inputs_ = layers.Input(shape=self.input_dim, name='inputs')
        return Model(inputs=inputs_, outputs=self.call(inputs_))
        # self._init_graph_network(inputs=self.input_layer,outputs=self.out)

    def call(self, input_data, **kwargs):
        x = self.encoder_layer1(input_data)
        x = self.encoder_layer2(x)
        x = self.latent(x)
        return x
