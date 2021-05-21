from keras.layers.core import Layer
import tensorflow as tf
from keras import backend as K


class LRN(Layer):
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, X, mask=None):
        b, ch, r, c = X._keras_shape
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = tf.zeros_like(X)  # make an empty tensor with zero pads along channel dimension
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :], input_sqr, extra_channels[:, half_n:2*half_n, :, :]], axis = 1)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
            scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):
    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, 1:, 1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
