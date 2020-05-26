import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class MaskPadding(tf.keras.layers.Layer):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def call(self, inputs, **kwargs):
        bos_mask = 1 - tf.reduce_sum(tf.one_hot([1, 2], self.vocab_size), axis=0)
        return bos_mask * inputs


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
        self.dense_proj_2 = layers.Dense(num_topic, activation=activation)
        self.dropout_1 = layers.Dropout(enc_drop)
        self.dropout_2 = layers.Dropout(enc_drop)

        self.dense_mean = layers.Dense(num_topic)
        self.dense_log_var = layers.Dense(num_topic)

    def call(self, inputs):
        x = self.dense_proj_1(inputs)
        x = self.dropout_1(x)
        x = self.dense_proj_2(x)
        x = self.dropout_2(x)
        mu_theta = self.dense_mean(x)
        logsigma_theta = self.dense_log_var(x)
        kl_theta = -0.5 * tf.reduce_mean(1 + logsigma_theta - tf.pow(mu_theta, 2) - tf.exp(logsigma_theta),
                                         axis=-1)
        return mu_theta, logsigma_theta, kl_theta


class Decoder(layers.Layer):
    def call(self, inputs, **kwargs):
        theta, beta = inputs
        res = tf.matmul(theta, beta)
        return tf.math.log(res + 1e-5)


class ETM:
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size,
                 theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5, seq_length=128):
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size  # embedding size for both vocabulary & topic
        self.enc_drop = enc_drop
        self.seq_length = seq_length

        ## topic_list
        self.topic_list = tf.constant([list(range(num_topics))])

        ## word embedding
        if train_embeddings:
            self.rho = tf.Variable(tf.random.normal((vocab_size, rho_size)))
        else:
            self.rho = tf.Variable(embeddings, trainable=False)

        ## topic embedding matrix
        self.alpha = tf.Variable(tf.random.normal((num_topics, rho_size)))

        ## vi encoder
        self.encoder = Encoder(num_topics, t_hidden_size, 'encoder', theta_act, enc_drop)

        ## vi decoder
        self.decoder = Decoder()

        ## sampling
        self.sampler = Reparameterize()

    def build(self):
        input_layer = layers.Input(batch_shape=(None, None), dtype=tf.int32)

        bows = tf.reduce_sum(tf.one_hot(input_layer, self.vocab_size), axis=1)
        bows = layers.Lambda(lambda x: x * (1 - tf.reduce_sum(tf.one_hot([1, 2], self.vocab_size), axis=0)))(bows)

        normal_bows = bows / tf.expand_dims(tf.reduce_sum(bows, axis=-1), -1)

        mu_theta, logsigma_theta, kl_theta = self.encoder(normal_bows)
        z = self.sampler([mu_theta, logsigma_theta])
        theta = layers.Softmax(axis=-1)(z)  # ( batch, num_topics )

        beta = tf.einsum('TE,VE->TV', self.alpha, self.rho)
        beta = layers.Softmax(axis=-1)(beta)

        lookup_matrix = self.decoder([theta, beta])  # (batch, num_vocab)

        recon_loss = - tf.reduce_sum(lookup_matrix * bows, axis=-1)
        loss = recon_loss + kl_theta
        # loss = tf.reduce_mean(loss)
        loss = tf.keras.layers.Activation('linear', dtype=tf.float32)(loss)
        self.model = tf.keras.Model(input_layer, [theta, normal_bows])
        self.model.add_loss(loss)

    def generate_topic_words(self):
        beta = tf.einsum('TE,VE->TV', self.alpha, self.rho)
        beta = layers.Softmax(axis=-1)(beta)
        represent_sort = tf.argsort(beta, direction='DESCENDING')
        represent_sort = represent_sort[:, :20].numpy()

        return represent_sort

#
# class ETM:
#     def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size,
#                  theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5, seq_length=128):
#         self.num_topics = num_topics
#         self.vocab_size = vocab_size
#         self.t_hidden_size = t_hidden_size
#         self.rho_size = rho_size  # embedding size for both vocabulary & topic
#         self.enc_drop = enc_drop
#         self.seq_length = seq_length
#
#         ## topic_list
#         self.topic_list = tf.constant([list(range(num_topics))])
#
#         ## word embedding
#         if train_embeddings:
#             self.rho = layers.Embedding(self.vocab_size, self.rho_size)
#         else:
#             self.rho = layers.Embedding(self.vocab_size, self.rho_size, weights=[embeddings], trainable=False)
#
#         ## topic embedding matrix
#         self.alpha = tf.Variable(tf.random.normal((num_topics, rho_size)))
#
#         ## vi encoder
#         self.encoder = Encoder(num_topics, t_hidden_size, 'encoder', theta_act, enc_drop)
#
#         ## vi decoder
#         self.decoder = Decoder()
#
#         ## sampling
#         self.sampler = Reparameterize()
#
#     def build(self):
#         input_layer = layers.Input(batch_shape=(None, self.seq_length), dtype=tf.int32)
#         input_mask = layers.Lambda(lambda x: K.cast(K.greater(x, 0), 'int32'), name="input_mask")(input_layer)
#
#         word_emb = self.rho(input_layer)
#         scores = layers.Dense(1, activation='linear')(word_emb)
#         scores = SoftmaxWithMask()(scores, mask=input_mask, axis=1)
#
#         document_embedding = tf.einsum('BTS,BTE->BT', scores, word_emb)
#
#         mu_theta, logsigma_theta, kl_theta = self.encoder(document_embedding)
#
#         z = self.sampler([mu_theta, logsigma_theta])
#         theta = layers.Softmax(axis=-1)(z)  # ( batch, num_topics )
#
#         beta = tf.einsum('TE,VE->TV', self.alpha, self.rho.embeddings)
#         beta = layers.Softmax(axis=-1)(beta)
#
#         lookup_matrix = self.decoder([theta, beta])  # (batch, num_vocab)
#         one_hot_input = tf.one_hot(input_layer, self.vocab_size)
#
#         recon_loss = - tf.einsum('BV,BTV->BT', lookup_matrix, one_hot_input)
#         recon_loss = recon_loss * tf.cast(input_mask, tf.float32)
#         recon_loss = tf.einsum('BT->B', recon_loss) / tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1)
#
#         loss = recon_loss + kl_theta
#         loss = tf.reduce_mean(loss)
#         loss = tf.keras.layers.Activation('linear')(loss)
#         self.model = tf.keras.Model(input_layer, theta)
#         self.model.add_loss(loss)
#
#     def generate_topic_words(self):
#         beta = tf.einsum('TE,VE->TV', self.alpha, self.rho.embeddings)
#         beta = layers.Softmax(axis=-1)(beta)
#         represent_sort = tf.argsort(beta, direction='DESCENDING')
#         represent_sort = represent_sort[:, :20].numpy()
#
#         return represent_sort


if __name__ == '__main__':
    m = ETM(num_topics=30, vocab_size=1000, t_hidden_size=128, rho_size=128, theta_act='relu')
    m.build()
    print(m.model.summary())
    a = np.random.randint(0, 1000, (64, 128))
    print(m.model.predict(a))

    print(m.generate_topic_words())
