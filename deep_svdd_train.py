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

MODEL_SAVE_DIR_PATH = os.path.abspath('pretrain_weight')
class DeepSVDDTrain:
    def __init__(self, args, svdd_model, train_dataset, test_dataset, pretrain_model=None):
        self.args = args
<<<<<<< HEAD
        #args를 fig변수로 바꾸고 fig는 json에서 load 한 뒤 값 담기
=======
>>>>>>> 92f00fc23286196eff4a1ecc958f9a746b40af92

        if pretrain_model:
            self.pretrain_model = pretrain_model
        self.svdd_model = svdd_model

        self.train_datasets = train_dataset
        self.test_datasets = test_dataset

<<<<<<< HEAD
        '''
        생각좀 해보기
        self.value = {'c':0.1,
                      'R': 0.0,
                      'nu': 0.1}
        '''
    #set_c ==> get_c로 바꾸기 get을 더 많이 씀
=======
>>>>>>> 92f00fc23286196eff4a1ecc958f9a746b40af92
    def set_c(self, eps=0.1):
        """Initializing the center for the hypersphere"""
        model = self.svdd_model.build_graph()
        if self.args['pretrain'] == True:
            model.load_weights(os.path.join(MODEL_SAVE_DIR_PATH, 'pretrain_ae.hdf5'))

<<<<<<< HEAD
            #temp 쓰지 말기
            z_list = []
            for (x_train_batch, _) in self.train_datasets:
                z = model.predict(x_train_batch)
                z_list.append(z)
            z_list = np.concatenate(z_list)
            c = z_list.mean(axis=0)
            # & ==> and로 바꾸기(속도가 더 빠름)
            c[(abs(c) < eps) & (c < 0)] = -eps  # avoid trivial solution that c = 0 is trivial solution
            c[(abs(c) < eps) & (c > 0)] = eps

        else:
            c = np.array(eps * self.svdd_model.latent_dim) # pretrain model 없을시 초기화

        return c

    def weight_loss(self, y_true, y_pred):
        self.dist_op = tf.reduce_sum(tf.square(y_pred - self.c), axis=-1) #evaluation 할 때 distance, score function 생성
=======
        # x_train = tf.data.Dataset.from_tensor_slices(self.x_train)
        # x_train = x_train.shuffle(buffer_size=1024).batch(64)

        z_list = []
        z_list_temp = []
        # new = np.array_split(x_train, 938, axis = 0)
        # for k, x in enumerate(x_train):
        #    z_list.append(model.predict(x))
        z_list = []
        for (x_train_batch, _) in self.train_datasets:
            z = model.predict(x_train_batch)
            z_list.append(z)
        z_list = np.concatenate(z_list)
        c = z_list.mean(axis=0)
        # c = z.mean(axis = 0)
        c[(abs(c) < eps) & (c < 0)] = -eps  # avoid trivial solution that c = 0 is trivial solution
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def radius_loss(self, y_true, y_pred):
        pass

    def weight_loss(self, y_true, y_pred):
        self.dist_op = tf.reduce_sum(tf.square(y_pred - self.c), axis=-1)
>>>>>>> 92f00fc23286196eff4a1ecc958f9a746b40af92
        score_op = self.dist_op - self.R ** 2
        penalty = tf.reduce_mean(tf.maximum(score_op, tf.zeros_like(score_op))) #tf.reduce_mean 추가
        loss_op = self.R ** 2 + (1 / self.nu) * penalty
        return loss_op


    def train_deep_svdd(self):
        model = self.svdd_model.build_graph()
        self.c = 0.1
        self.R = 0.0
        self.nu = 0.1
        if self.args['pretrain']:
            model.load_weights(os.path.join(MODEL_SAVE_DIR_PATH, 'pretrain_ae.hdf5'))
            self.c = self.set_c()  # need to compare between inference time and laod time

        """training code"""
        wd = self.args['weight_decay']
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        # loss_object = self.weight_loss()
        # optimizer = tf.keras.optimizers.Adam()
        optimizer = tfa.optimizers.AdamW(weight_decay=wd)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        epochs = self.args['epochs']
        for epoch in range(epochs):
            print(f"Start of epoch {epoch + 1}")

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_datasets):
                with tf.GradientTape() as tape:
                    reconstructed = model(x_batch_train)
                    # Compute reconstruction loss
                    loss = self.weight_loss(y_batch_train, reconstructed)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                #                grads = tape.gradient(loss, model)
                #                optimizer.apply_gradients(zip(grads, model))

                train_loss(loss)
                if (step + 1) % 50 == 0:
                    print(f"step {step + 1}: mean loss = {train_loss.result():.4f}")
                if (epoch + 1) % 5 == 0:
<<<<<<< HEAD
                    self.R = self.get_R()

        return model, self.R

    def get_R(self):
        return np.quantile(np.sqrt(self.dist_op.numpy()), 1 - self.nu)

=======
                    self.R = self.get_R(self.dist_op, self.nu)

                # if step % 10 == 0:
        return model, self.R

    def get_R(self):
        print(self.dist_op)
        print(self.dist_op.numpy())
        return np.quantile(np.sqrt(self.dist_op.numpy()), 1 - self.nu)

    def deep_svdd_train(self):
        pass

>>>>>>> 92f00fc23286196eff4a1ecc958f9a746b40af92
    def eval_step(self):
        pass
        # prediction =