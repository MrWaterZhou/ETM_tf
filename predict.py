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
parser.add_argument('--corpus',type=str)
parser.add_argument('--weight_path', type=str, default='./results', help='path to save results')
parser.add_argument('--emb_path',type=str)
parser.add_argument('--vocab_path', type=str, default=None)
parser.add_argument('--predefine_path', type=str, default=None)

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu',
                    help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')

### pred
parser.add_argument('--batch_size', type=int, default=256)

args = parser.parse_args()


def load_dataset(filenames, batch_size):
    if not isinstance(filenames, list):
        filenames = [filenames]

    def parse(line):
        line = tf.strings.split(line)
        x = tf.strings.to_number(line, tf.float32)
        return x

    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, (None,))
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
    vocab_set = set(vocab)
    predefine_topics = [x.strip().split(' ') for x in
                        open(args.predefine_path, 'r').readlines()] if args.predefine_path is not None else []

    if args.emb_path is not None:
        vectors = {}
        with open(args.emb_path, 'r') as f:
            for l in f:
                line = l.split()
                word = line[0]
                if word in vocab_set:
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

        idx = list(range(len(vocab)))
        np.random.shuffle(idx)

        topic_embeddings = embeddings[idx[:args.num_topics]]

        for i, topic_words in enumerate(predefine_topics):
            tmp_emb = np.zeros(args.rho_size)
            for word in topic_words:
                tmp_emb += vectors[word]
            topic_embeddings[i] = tmp_emb / len(topic_words)
        topic_embeddings = np.float32(topic_embeddings)

    else:
        embeddings = None
        topic_embeddings = None

    # build model
    etm = ETM(num_topics=args.num_topics, rho_size=args.rho_size, theta_act=args.theta_act,
              train_embeddings=args.train_embeddings, embeddings=embeddings, topic_embeddings=topic_embeddings,
              enc_drop=args.enc_drop,
              vocab_size=len(vocab), t_hidden_size=args.t_hidden_size)
    input_layer = tf.keras.layers.Input(batch_shape=(None, None), dtype=tf.int32)
    model = tf.keras.Model(input_layer, etm(input_layer))
    model.load_weights(args.weight_path)

    # loading data
    data = load_dataset(args.data_path, args.batch_size)
    corpus = open(args.corpus, 'r').readlines()

    # start predict
    topic_rep = etm.generate_topic_words()
    topic_represent = [[vocab[i] for i in x] for x in topic_rep]
    theta = model.predict(data)  # theta (batch, num_topics)
    print(np.argmax(theta[:10],axis=-1))

    f = open('topic_result.txt','w')
    for i, th in enumerate(theta):
        row = corpus[i].strip()
        topics = np.argsort(th)[::-1]
        for topic in topics:
            if th[int(topic)] > 0.05:
                topic_re = topic_represent[int(topic)]
                tmp = "corpus:{}\n topic:{}\n pred:{}\n".format(''.join(row), ','.join(topic_re), th[int(topic)])
                print(tmp)
                res = "{}\t{}\t{}\n".format(''.join(row), ','.join(topic_re), th[int(topic)])
                f.write(res)
    f.close()




