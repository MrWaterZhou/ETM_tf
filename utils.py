from tensorflow.keras.preprocessing.text import text_to_word_sequence
import tensorflow as tf
import re
import tensorflow_datasets as tfds


def load_vocab(filename):
    with open(filename, 'r') as f:
        vocab = [word.strip() for word in f]
    return vocab


def _fake_gen():
    while True:
        yield 0


class EngDataUtil:
    def __init__(self, vocab_path):
        self.vocab_list = load_vocab(vocab_path)
        self.tokenizer = tfds.features.text.TokenTextEncoder(self.vocab_list)
        self.vocab_size = self.tokenizer.vocab_size
        # self.pat = re.compile('#[0-9]{3}')

    def encode(self, text_tensor):
        encoded_text = self.tokenizer.encode(text_tensor.numpy())
        return encoded_text

    def encode_map_fn(self, text):
        ids = tf.py_function(self.encode, inp=[text], Tout=(tf.int32))
        ids.set_shape([None])
        return ids

    def load_dataset(self, filenames, batch_size):
        if not isinstance(filenames, list):
            filenames = [filenames]
        dataset = tf.data.TextLineDataset(filenames)
        dataset = dataset.map(self.encode_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size, [None])
        dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_generator(_fake_gen, tf.int32))).shuffle(64)
        return dataset

if __name__=='__main__':
    du = EngDataUtil('vocab_new.txt')
    ds = du.load_dataset('data/eng_sample.txt')
    for i,j in ds:
        print(i)
        break
