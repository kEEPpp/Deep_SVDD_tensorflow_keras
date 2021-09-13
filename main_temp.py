import json
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
from tensorflow.keras.models import Model

from Data.Data_Loader import Data_Loader
from Model.simple_model import Pretrain_AutoEncoder
from Model.simple_model import DeepSVDD
from deep_svdd_train import DeepSVDDTrain
from pretrain import PreTrain

def convert_str_to_bool(file):
    for key, value in file.items():
        if value == "True":
            file[key] = True
    return file

def main():
    json_path = os.path.join(os.path.abspath('config'), 'config.json')
    with open(json_path, 'r') as f:
        fig = json.load(f)
        fig = convert_str_to_bool(fig)


if __name__ == '__main__':
    main()

