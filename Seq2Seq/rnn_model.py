import tensorflow as tf
from ops import MLP

class rnn_encoder:
    def __init__(self, name, layerSize, hiddenSize, embedSize, sourceVocSize, targetVocSize, rnnType='lstm'):
        self.name = name
        self.layerSize = layerSize
        self.hiddenSize = hiddenSize
        self.embedSize = embedSize
        self.sourceVocSize = sourceVocSize
        self.targetVocSize = targetVocSize
        self.rnnType = rnnType

    def __call__(self, seqVec, seqNums, batchSize):
        embeddingMat_name = "embed_name"
        logit_name = "logit"
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.rnnType == "lstm":
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenSize)
            if self.rnnType == "rnn":
                cell = tf.nn.rnn_cell.BasicRNNCell(self.hiddenSize)
            if self.rnnType == "gru":
                cell = tf.nn.rnn_cell.GRUCell(self.hiddenSize)
            cells = tf.nn.rnn_cell.MultiRNNCell([cell] * self.layerSize)
            init_state = cells.zero_state(batchSize, tf.float32)
            embeddingMat = tf.get_variable(embeddingMat_name, [self.sourceVocSize, self.embedSize], initializer=tf.truncated_normal_initializer(stddev=0.02))
            seqVec = tf.nn.embedding_lookup(embeddingMat, seqVec)
            _, states = tf.nn.dynamic_rnn(cells, seqVec, seqNums, init_state)
            # logit = MLP(logit_name, outputs[:, -1, :], self.hiddenSize, self.targetVocSize)
            # prob = tf.nn.softmax(logit)
            return states

class rnn_decoder:
    def __init__(self, name, layerSize, hiddenSize, embedSize, targetVocSize, rnnType='lstm'):
        self.name = name
        self.layerSize = layerSize
        self.hiddenSize = hiddenSize
        self.embedSize = embedSize
        self.targetVocSize = targetVocSize
        self.rnnType = rnnType

    def __call__(self, seqVec, seqNums, batchSize, preState=None):
        embeddingMat_name = "embed_name"
        logit_name = "logit"
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.rnnType == "lstm":
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenSize)
            if self.rnnType == "rnn":
                cell = tf.nn.rnn_cell.BasicRNNCell(self.hiddenSize)
            if self.rnnType == "gru":
                cell = tf.nn.rnn_cell.GRUCell(self.hiddenSize)
            cells = tf.nn.rnn_cell.MultiRNNCell([cell] * self.layerSize)
            init_state = cells.zero_state(batchSize, tf.float32)
            embeddingMat = tf.get_variable(embeddingMat_name, [self.targetVocSize, self.embedSize], initializer=tf.truncated_normal_initializer(stddev=0.02))
            seqVec = tf.nn.embedding_lookup(embeddingMat, seqVec)
            if preState != None:
                outputs, states = tf.nn.dynamic_rnn(cells, seqVec, seqNums, preState)
            else:
                outputs, states = tf.nn.dynamic_rnn(cells, seqVec, seqNums, init_state)
            outputs = tf.reshape(outputs, [-1, self.hiddenSize])
            logits = MLP(logit_name, outputs, self.hiddenSize, self.targetVocSize)
            probs = tf.nn.softmax(logits)
            return probs, states, init_state





