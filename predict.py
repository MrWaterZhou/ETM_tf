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
parser.add_argument('--corpus', type=str)
parser.add_argument('--weight_path', type=str, default='./results', help='path to save results')
parser.add_argument('--vocab_path', type=str, default=None)

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu',
                    help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')

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



if __name__ == '__main__':
    vocab = [x.strip() for x in open(args.vocab_path, 'r').readlines()]
    vocab_dic = {x:i for i,x in enumerate(vocab)}

    # build model
    etm = ETM(num_topics=args.num_topics, rho_size=args.rho_size, theta_act=args.theta_act,
              train_embeddings=1, embeddings=None, topic_embeddings=None,
              enc_drop=0,
              vocab_size=len(vocab), t_hidden_size=args.t_hidden_size)
    input_layer = tf.keras.layers.Input(batch_shape=(None, None), dtype=tf.int32)
    model = tf.keras.Model(input_layer, etm(input_layer))
    model.load_weights(args.weight_path)
    print(model.summary())

    # loading data
    corpus = open(args.corpus, 'r').readlines()
    data = [[vocab_dic[word] for word in x.strip().split() if word in vocab_dic] for x in corpus]
    data = tf.keras.preprocessing.sequence.pad_sequences(data,padding='post')

    # # start predict
    topic_rep = etm.generate_topic_words()
    topic_represent = [[vocab[i] for i in x] for x in topic_rep]
    theta = model.predict(data, batch_size=256)  # theta (batch, num_topics)
    print(np.argmax(theta[:10], axis=-1))

    f = open('topic_result.txt', 'w')
    for th,row in zip(theta,corpus):
        topics = np.argsort(th)[::-1][:10]
        for topic in topics:
            if th[int(topic)] > 0.05:
                topic_re = topic_represent[int(topic)]
                tmp = "corpus:{}\n topic:{}\n pred:{}\n".format(''.join(row), ','.join(topic_re), th[int(topic)])
                print(tmp)
                res = "{}\t{}\t{}\n".format(''.join(row), ','.join(topic_re), th[int(topic)])
                f.write(res)
    f.close()
