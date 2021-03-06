import os
import collections
from six.moves import cPickle
import numpy as np

def create_vocab(text):
    # create vocabulary dict
    # e.g. from text "gallahad" -> {'a': 0, 'h': 2, 'l': 1, 'g': 3, 'd': 4}
    counter = collections.Counter(text)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocab_size = len(chars)
    vocab = dict(zip(chars, range(vocab_size)))
    return vocab, chars

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        labels_file = os.path.join(data_dir, "labels.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")
        target_file = os.path.join(data_dir, "target.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text and laebls file")
            self.preprocess(input_file, labels_file, vocab_file, tensor_file, target_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file, target_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, labels_file, vocab_file, tensor_file, target_file):
        with open(input_file, "r") as f:
            data = f.read()
        with open(labels_file, "r") as f:
            labels = f.read()

        self.vocab, self.chars = self.create_vocab(data)
        self.vocab_size = len(self.chars)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        # transforms text in a numpy array of integers
        # e.g. from "gallahad" -> array([3, 0, 1, 1, 0, 2, 0, 4])
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

        self.targets, self.classes = self.create_vocab(labels)
        self.tensor = np.array(list(map(self.targets.get, labels)))
        np.save(target_file, self.targets)

    def load_preprocessed(self, vocab_file, tensor_file, target_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.targets = np.load(target_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        self.targets = self.targets[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = self.targets
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
