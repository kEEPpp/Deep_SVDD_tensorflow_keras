from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Pretrain_AutoEncoder(Model):
    def __init__(self, latent_dim: int, input_dim: int):
        super(Pretrain_AutoEncoder, self).__init__()
        #encoder
        self.encoder_layer1 = layers.Dense(500, activation='relu', name='encoder1')
        self.encoder_layer2 = layers.Dense(350, activation='relu', name='encoder2')
        self.latent = layers.Dense(latent_dim, activation='relu', name='latent')

        #decoder
        self.decoder_layer1 = layers.Dense(350, activation='relu', name='decoder1')
        self.decoder_layer2 = layers.Dense(500, activation='relu', name='decoder2')
        self.outputs = layers.Dense(784, activation='relu', name='outputs')

        #parameter
        self.input_dim = input_dim
        self.latent_dim = latent_dim

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
    def __init__(self, latent_dim: int, input_dim: int):
        super(DeepSVDD, self).__init__()
        #encoder
        self.encoder_layer1 = layers.Dense(500, activation='relu', name='encoder1')
        self.encoder_layer2 = layers.Dense(350, activation='relu', name='encoder2')
        self.latent = layers.Dense(latent_dim, activation='relu', name='latent')

        #parameter
        self.latent_dim = latent_dim
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


class Pretrain_Autoencoder_HAICon2021(Model):
    def __init__(self, latent_dim: int, input_dim: int):
        super(Pretrain_Autoencoder_HAICon2021, self).__init__()
        #encoder
        self.encoder_layer1 = layers.GRU(500, activation='relu', name='encoder1')
        self.encoder_layer2 = layers.GRU(350, activation='relu', name='encoder2')
        self.latent = layers.Dense(latent_dim, activation='relu', name='latent')

        #decoder
        self.decoder_layer1 = layers.Dense(350, activation='relu', name='decoder1')
        self.decoder_layer2 = layers.Dense(500, activation='relu', name='decoder2')
        self.outputs = layers.Dense(784, activation='relu', name='outputs')

        #parameter
        self.input_dim = input_dim
        self.latent_dim = latent_dim

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