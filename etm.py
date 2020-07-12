import tensorflow as tf
# import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
# from tensorflow_core.python import GlorotUniform, GlorotNormal


class SoftmaxWithMask(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor, mask, axis):
        mask = tf.cast(mask, inputs.dtype)
        mask = tf.expand_dims(mask, -1)
        masked_logits = tf.multiply(tf.exp(inputs), mask)
        softmax = tf.divide(masked_logits, tf.reduce_sum(masked_logits, axis, keepdims=True) + 1e-5)
        return softmax


class Reparameterize(layers.Layer):
    def call(self, inputs, training=None):
        mu = inputs[0]
        logvar = inputs[1]
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        if training is None:
            training = K.learning_phase()
        if training:
            std = tf.exp(0.5 * logvar)
            eps = tf.keras.backend.random_normal(shape=(batch, dim))
            return tf.add(tf.multiply(eps, std), mu)
        else:
            return mu


class Encoder(layers.Layer):
    def __init__(self, num_topic, t_hidden_size, name, activation, enc_drop, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj_1 = layers.Dense(t_hidden_size, activation=activation)
        self.dense_proj_2 = layers.Dense(t_hidden_size, activation=activation)
        self.dropout_2 = layers.Dropout(enc_drop)

        self.dense_mean = layers.Dense(num_topic)
        self.dense_log_var = layers.Dense(num_topic)

    def call(self, inputs):
        x = self.dense_proj_1(inputs)
        x = self.dense_proj_2(x)
        x = self.dropout_2(x)
        mu_theta = self.dense_mean(x)
        logsigma_theta = self.dense_log_var(x)
        kl_theta = -0.5 * tf.reduce_sum(1 + logsigma_theta - tf.pow(mu_theta, 2) - tf.exp(logsigma_theta),
                                        axis=-1)
        return mu_theta, logsigma_theta, tf.maximum(kl_theta, 5.0)


class EncoderDense(layers.Layer):
    def __init__(self, num_topic, t_hidden_size, vocab_size, enc_drop, name, activation, **kwargs):
        super(EncoderDense, self).__init__(name=name, **kwargs)
        self.embedding_hidden = layers.Embedding(input_dim=vocab_size, output_dim=t_hidden_size)
        self.dense_proj_1 = layers.Dense(t_hidden_size, activation=activation)
        self.dropout_2 = layers.Dropout(enc_drop)
        self.dense_mean = layers.Dense(num_topic)
        self.dense_log_var = layers.Dense(num_topic)
        self.mask_layer = layers.Lambda(lambda x: K.expand_dims(K.cast(K.greater(x, 0), tf.float32)), name="input_mask")

    def call(self, inputs):
        x = self.embedding_hidden(inputs)
        x = self.dense_proj_1(x)
        x = self.dropout_2(x)
        input_mask = self.mask_layer(inputs)
        x = tf.reduce_sum(x * input_mask, axis=1) / tf.reduce_sum(input_mask, axis=1)
        mu_theta = self.dense_mean(x)
        logsigma_theta = self.dense_log_var(x)
        kl_theta = -0.5 * tf.reduce_sum(1 + logsigma_theta - tf.pow(mu_theta, 2) - tf.exp(logsigma_theta),
                                        axis=-1)
        return mu_theta, logsigma_theta, kl_theta


class Decoder(layers.Layer):
    def call(self, inputs, **kwargs):
        theta, beta = inputs
        res = tf.matmul(theta, beta)
        return tf.math.log(res + 1e-5)


class ETM(tf.keras.layers.Layer):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size,
                 theta_act, embeddings=None, topic_embeddings=None, train_embeddings=True, enc_drop=0.5, seq_length=128,
                 name='etm', **kwargs):
        super(ETM, self).__init__(name=name, **kwargs)

        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size  # embedding size for both vocabulary & topic
        self.enc_drop = enc_drop
        self.seq_length = seq_length

        w_init = tf.keras.initializers.RandomUniform(-1, 1)
        if train_embeddings:
            self.rho = tf.Variable(w_init(shape=(vocab_size, rho_size)), trainable=True)
            self.alpha = tf.Variable(w_init(shape=(num_topics, rho_size)), trainable=True)
        else:
            self.rho = tf.Variable(embeddings, trainable=False)
            self.alpha = tf.Variable(topic_embeddings, trainable=True)

        ## vi encoder
        # self.encoder = Encoder(num_topics, t_hidden_size, 'encoder', theta_act, enc_drop)
        self.encoder = EncoderDense(num_topics, t_hidden_size, vocab_size, enc_drop, 'encoder', theta_act)

        ## vi decoder
        self.decoder = Decoder()

        ## sampling
        self.sampler = Reparameterize()

    def call(self, input_ids, **kwargs):
        mu_theta, logsigma_theta, kl_theta = self.encoder(input_ids)

        print(mu_theta, logsigma_theta, kl_theta)
        z = self.sampler([mu_theta, logsigma_theta])
        theta = layers.Softmax(axis=-1)(z)  # (batch, num_topic)

        beta = tf.einsum('TE,VE->TV', self.alpha, self.rho)  # (num_topic, num_vocab)
        beta = layers.Softmax(axis=-1)(beta)

        lookup_matrix = self.decoder([theta, beta])  # (batch, num_vocab)
        lookup_matrix = tf.einsum('BV->VB',lookup_matrix) # (num_vocab, batch')
        recon_loss = tf.nn.embedding_lookup(lookup_matrix, input_ids) # (batch, seq_size, batch')
        recon_loss = self.encoder.mask_layer(input_ids) * recon_loss
        recon_loss = tf.einsum('BSN->BN',recon_loss)
        recon_loss = - tf.linalg.diag_part(recon_loss)

        loss = tf.reduce_mean(recon_loss) + tf.reduce_mean(kl_theta)

        self.add_loss(loss)
        self.add_metric(recon_loss, name='recon_loss', aggregation='mean')
        self.add_metric(kl_theta, name='kl_theta', aggregation='mean')

        return theta

    def generate_topic_words(self):
        beta = tf.einsum('TE,VE->TV', self.alpha, self.rho)
        beta = layers.Softmax(axis=-1)(beta)
        represent_sort = tf.argsort(beta, direction='DESCENDING')
        represent_sort = represent_sort[:, :10].numpy()

        return represent_sort


if __name__ == '__main__':
    m = ETM(num_topics=30, vocab_size=1000, t_hidden_size=128, rho_size=128, theta_act='relu')
    input_ids = layers.Input(batch_shape=(None, None), dtype=tf.int32)

    model = tf.keras.Model(input_ids, m(input_ids))

    print(model.summary())
    a = tf.constant([[1,2,3],[1,0,0]])
    b = tf.constant([[1],[2]])
    print(model.predict(a))
    print(model.predict(b))

    # batch = 2
    # num_topic = 30
    # num_vocab = 1000
    #
    # theta = tf.random.uniform((batch, num_topic))
    # beta = tf.random.uniform((num_topic, num_vocab))
    #
    # ids = tf.constant([[4, 7, 0], [1, 6, 9]])
    # bows = tf.reduce_sum(tf.one_hot(ids, num_vocab), 1)
    #
    # lookup_matrix = tf.math.log(1e-5 + tf.matmul(theta, beta))
    # x0 = tf.reduce_sum(lookup_matrix * bows, axis=-1)
    #
    # lookup_matrix_T = tf.einsum('BV->VB',lookup_matrix)
    # x1 = tf.nn.embedding_lookup(lookup_matrix_T,ids)
    # x1 = tf.einsum('BSN->BN',x1)
    # x1 = tf.linalg.diag_part(x1)
    #
    # print(x0)
    # print(x1)
    #
    #
    #
