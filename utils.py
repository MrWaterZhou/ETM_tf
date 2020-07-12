from tensorflow.keras.preprocessing.text import text_to_word_sequence
import tensorflow as tf
import re


def load_vocab(filename):
    with open(filename, 'r') as f:
        vocab = {word.strip(): i for i, word in enumerate(f)}
    return vocab

def _fake_gen():
    while True:
        yield 0

class EngDataUtil:
    def __init__(self, vocab_path):
        self.vocab = load_vocab(vocab_path)
        self.vocab_size = len(self.vocab)
        self.pat = re.compile('#[0-9]{3}')

    def text_to_ids(self, text):
        text = self.pat.sub('', text)
        words = text_to_word_sequence(text)
        ids = [self.vocab[word] for word in words if word in self.vocab]
        return ids

    def ids_to_bows(self, ids: list):
        bows = [0] * self.vocab_size
        for id in ids:
            bows[id] += 1
        return bows

    def create_batch(self, texts):
        ids_list = [self.text_to_ids(text) for text in texts]
        bows_list = [self.ids_to_bows(ids) for ids in ids_list]
        return ids_list, bows_list

    def create_dataset(self, source_file, target_file):
        source = open(source_file, 'r')
        target = open(target_file, 'w')
        batch_size = 10240

        while True:
            texts = source.readlines(batch_size)
            if len(texts) == 0:
                break
            ids_list, bows_list = self.create_batch(texts)
            tmp = [' '.join([str(i) for i in ids]) + '@' + ' '.join([str(i) for i in bows]) for ids, bows in
                   zip(ids_list, bows_list)]
            for t in tmp:
                target.write(t + '\n')

        source.close()
        target.close()

    def load_dataset(self, filenames, batch_size):
        if not isinstance(filenames, list):
            filenames = [filenames]

        def parse(line):
            res = tf.strings.split(line, sep='@')
            label = tf.strings.split(res[1])
            line = tf.strings.split(res[0])

            line = tf.strings.to_number(line, tf.int32)
            label = tf.strings.to_number(label, tf.float16)
            return line, label

        dataset = tf.data.TextLineDataset(filenames)
        dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size, ([None], [None]))
        dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_generator(_fake_gen, tf.int32))).shuffle(64)
        return dataset
