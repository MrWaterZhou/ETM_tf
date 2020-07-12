import argparse
from utils import EngDataUtil, text_to_word_sequence

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--vocab_path', type=str, default=None)
parser.add_argument('--source_path', type=str)
parser.add_argument('--target_path', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    if args.vocab_path is None:
        word_count = {}
        with open(args.source_path, 'r') as f:
            for line in f:
                words = text_to_word_sequence(line)
                for word in word:
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1

        sorted_word = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        args.vocab_path = 'vocab.txt'
        vocab = open(args.vocab_path, 'w')
        vocab.write('<PAD>\n')
        for pair in sorted_word[:32000]:
            vocab.write(pair[0] + '\n')
        vocab.close()

    du = EngDataUtil(args.vocab_path)
    du.create_dataset(args.source_path, args.target_path)
