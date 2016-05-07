import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, rnn
from tensorflow.models.rnn import seq2seq

import numpy as np

test_text = '500 g	Nudeln (Rigatoni o. ä.)\n250 g	Kirschtomate(n)\n1/2 Glas	Oliven, schwarz, ohne Kerne'

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        # create tensorflow placeholder
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        # Initial state of the cell memory.
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # create namespace for shareable variables (variable name = "rnnlm/softmax_w")
        with tf.variable_scope('rnnlm'):
            # create (or get) a variable with shape [rnn_size, vocab_size]
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                # preparing dense representation of the data in a embedding matrix
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # rnn network
        outputs, last_state = seq2seq.rnn_decoder(inputs,
                                                  self.initial_state,
                                                  cell,
                                                  loop_function=loop if infer else None,
                                                  scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        # last layer (like fully connected nn)
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        # activation function of the last layer
        self.probs = tf.nn.softmax(self.logits)

        # loss function
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)

        # training function
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.final_state = last_state

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret


class ClassificationModel():
    def __init__(self, batch_size=50, seq_length=20, rnn_size=64, layer_depth=2, infer=False):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rnn_size = rnn_size
        self.layer_depth = layer_depth


        with tf.variable_scope('rnnlm'):

            # Input #
            #########

            # create tensorflow placeholder
            self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
            self.targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])

            # define network
            with tf.device("/cpu:0"):
                # preparing dense representation of the data in a embedding matrix
                embedding = tf.get_variable("embedding", [self.vocab_size, self.rnn_size])
                inputs = tf.split(1, self.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            # Hidden #
            ##########

            lstm = rnn_cell.BasicLSTMCell(self.rnn_size)
            self.stacked_lstm = rnn_cell.MultiRNNCell([lstm] * self.layer_depth)
            # Initial state of the cell memory.
            self.initial_state = self.stacked_lstm.zero_state(self.batch_size, tf.float32)

            outputs, last_state = rnn.rnn(self.stacked_lstm, inputs,
                                          initial_state=self.initial_state,
                                          scope='rnnlm')



        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        # create tensorflow placeholder
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        # Initial state of the cell memory.
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # create namespace for shareable variables (variable name = "rnnlm/softmax_w")
        with tf.variable_scope('rnnlm'):
            # create (or get) a variable with shape [rnn_size, vocab_size]
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                # preparing dense representation of the data in a embedding matrix
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # rnn network
        outputs, last_state = seq2seq.rnn_decoder(inputs,
                                                  self.initial_state,
                                                  cell,
                                                  loop_function=loop if infer else None,
                                                  scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        # last layer (like fully connected nn)
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        # activation function of the last layer
        self.probs = tf.nn.softmax(self.logits)

        # loss function
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)

        # training function
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.final_state = last_state

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret

    def test(self, sess, chars, vocab, num=200, prime=test_text, sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret

