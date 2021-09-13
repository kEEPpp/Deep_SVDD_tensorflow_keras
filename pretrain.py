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
#from tensorflow.keras import losses
from tensorflow.keras.models import Model

#MODEL_SAVE_DIR_PATH = os.path.join(os.path.abspath('Train'), 'pretrain_weight')  # '.\\pretrain_weight\\'
MODEL_SAVE_DIR_PATH = os.path.abspath('pretrain_weight')

class PreTrain:
    def __init__(self, args : dict, ae_model, train_dataset, test_dataset):
        self.args = args
        self.ae_model = ae_model

        self.train_datasets = train_dataset
        self.test_datasets = test_dataset

    def start_check(self):
        if not os.path.exists('pretrain_weight'):
            os.mkdir('pretrain_weight')

    def pretrain_ae(self):
        ae = self.ae_model.build_graph()
        loss_object = tf.keras.losses.MeanSquaredError()  # SVDD
        optimizer = tf.keras.optimizers.Adam()

        ae.compile(optimizer=optimizer, loss=loss_object, metrics=['mae'])
        hist = ae.fit(self.train_datasets, epochs=self.args['epochs'])

        encoder_part = Model(ae.get_layer('inputs').input, ae.get_layer('latent').output)
        self.save_weights_pretrain(encoder_part)

        return hist

    def save_weights_pretrain(self, model):
        """학습된 AutoEncoder 가중치를 DeepSVDD모델에 Initialize해주는 함수"""
        # c = self.set_c()
        print(f"Saved to {os.path.join(MODEL_SAVE_DIR_PATH, 'pretrain_ae.hdf5')}")
        model.save_weights(os.path.join(MODEL_SAVE_DIR_PATH, 'pretrain_ae.hdf5'))  # model save

