import os

import matplotlib.pyplot as plt
import numpy as np

from Data.Data_Loader import Data_Loader

MODEL_SAVE_DIR_PATH = os.path.abspath('pretrain_weight')


class Test:
    def __init__(self, config, svdd, pretrain, value):
        self.config = config
        self.svdd = svdd
        self.pretrain = pretrain
        self.value = value
        self.data_loader = Data_Loader(self.config)

    def pretrain_evaluation(self):
        train_dataset, test_dataset = self.data_loader.pretrain_mnist()
        self.pretrain.buld()
        self.pretrain.load_weights((os.path.join(MODEL_SAVE_DIR_PATH, 'pretrain_ae.hdf5')))

        '''
        pred = self.pretrain.predict(test_dataset)
        pred = pred.reshape(-1,28,28)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(pred[0])
        ax[1].imshow(pred[1])
        plt.show()
        '''
        return self.pretrain.predict(test_dataset)  # shape 확인

    def dist(self, pred):
        return np.sqrt(np.sum(((self.value['c'] - pred) ** 2), axis=-1))

    def score(self, pred):
        return self.dist(pred) - self.value['R']

        #scores_normal_index = np.where(scores > 0)

    def svdd_evaluation(self, is_train=False):
        train_dataset, test_dataset = self.data_loader.svdd_mnist()
        if is_train:
            self.svdd.predict(train_dataset)
        else:
            pre = self.svdd.predict(test_dataset)


    def plotting(self):
        pass
