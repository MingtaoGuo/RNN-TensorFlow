import tensorflow as tf
from ops import MLP, preprocess
from inception_v2 import inception_v2, inception_v2_arg_scope
import tensorflow.contrib.slim as slim


class rnn_decoder:
    def __init__(self, name, layerSize, hiddenSize, embedSize, targetVocSize, rnnType='lstm'):
        self.name = name
        self.layerSize = layerSize
        self.hiddenSize = hiddenSize
        self.embedSize = embedSize
        self.targetVocSize = targetVocSize
        self.rnnType = rnnType

    def __call__(self, imgs, seqVec, seqNums, batchSize):
        arg_scope = inception_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            _, feat = inception_v2(preprocess(imgs), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            img_inputs = MLP("img_embedding", feat, 1024, self.embedSize)
            img_inputs = tf.expand_dims(img_inputs, axis=1)
            if self.rnnType == "lstm":
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenSize)
            if self.rnnType == "rnn":
                cell = tf.nn.rnn_cell.BasicRNNCell(self.hiddenSize)
            if self.rnnType == "gru":
                cell = tf.nn.rnn_cell.GRUCell(self.hiddenSize)
            cells = tf.nn.rnn_cell.MultiRNNCell([cell] * self.layerSize)
            if batchSize > 1:
                cells = tf.nn.rnn_cell.DropoutWrapper(cells, input_keep_prob=0.7, output_keep_prob=0.7, state_keep_prob=0.7)
            init_state = cells.zero_state(batchSize, tf.float32)
            _, img_states = tf.nn.dynamic_rnn(cells, img_inputs, initial_state=init_state)
            embeddingMat = tf.get_variable("embeddingMat", [self.targetVocSize, self.embedSize], initializer=tf.truncated_normal_initializer(stddev=0.08))
            seqVec = tf.nn.embedding_lookup(embeddingMat, seqVec)
            if batchSize == 1:
                #Test phase
                outputs, states = tf.nn.dynamic_rnn(cells, seqVec, initial_state=init_state)
                outputs = tf.reshape(outputs, [-1, self.hiddenSize])
                logits = MLP("logits", outputs, self.hiddenSize, self.targetVocSize)
                probs = tf.nn.softmax(logits)
                wordVal = tf.argmax(probs, axis=1)
                return probs, wordVal, states, img_states, init_state
            else:
                #Training phase
                outputs, _ = tf.nn.dynamic_rnn(cells, seqVec, seqNums, img_states)
                outputs = tf.reshape(outputs, [-1, self.hiddenSize])
                logits = MLP("logits", outputs, self.hiddenSize, self.targetVocSize)
                probs = tf.nn.softmax(logits)
                return probs





