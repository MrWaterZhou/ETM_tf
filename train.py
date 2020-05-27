import argparse

from etm import ETM
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--emb_path', type=str, default=None,
                    help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')
parser.add_argument('--vocab_path', type=str, default=None)

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu',
                    help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')

args = parser.parse_args()


def load_dataset(filenames, batch_size):
    if not isinstance(filenames, list):
        filenames = [filenames]

    def parse(line):
        line = tf.strings.split(line).to_tensor()
        x = tf.strings.to_number(line, tf.int32)
        return x

    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(2048)
    dataset = dataset.padded_batch(batch_size)
    return dataset


class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, etm: ETM, vocab: list, save_path: str):
        self.etm = etm
        self.vocab = vocab
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            topic_rep = self.etm.generate_topic_words()
            topic_words = [[self.vocab[i] for i in x] for x in topic_rep]
            for i, topic in enumerate(topic_words):
                print('topic {}:{}\n'.format(i, ', '.join(topic)))
            self.model.save_weights(os.path.join(self.save_path, '{}_weight'.format(epoch)))


if __name__ == '__main__':

    vocab = [x.strip() for x in open(args.vocab_path, 'r').readlines()]

    if args.emb_path is not None:
        vectors = {}
        with open(args.emb_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                if word in vocab:
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
        embeddings = np.zeros((len(vocab), args.rho_size))
        words_found = 0
        for i, word in enumerate(vocab):
            try:
                embeddings[i] = vectors[word]
                words_found += 1
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(args.rho_size,))
        embeddings = np.float32(embeddings)
    else:
        embeddings = None

    # build model
    etm = ETM(num_topics=args.num_topics, rho_size=args.rho_size, theta_act=args.theta_act,
              train_embeddings=args.train_embeddings, embeddings=embeddings, enc_drop=args.enc_drop,
              vocab_size=len(vocab), t_hidden_size=args.t_hidden_size)
    input_layer = tf.keras.layers.Input(batch_shape=(None, 128), dtype=tf.int32)
    model = tf.keras.Model(input_layer, etm(input_layer))
    print(model.summary())

    # loading data
    data = load_dataset(args.data_path, args.batch_size)

    # start training
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    vis = VisCallback(etm, vocab, args.save_path)

    optimizer = tf.keras.optimizers.Adam(args.lr)
    model.compile(optimizer=optimizer, loss=None)
    model.fit(data, epochs=args.epochs, batch_size=args.batch_size, callbacks=[vis])
