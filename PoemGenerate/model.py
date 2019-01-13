import tensorflow as tf
from ops import MLP

def rnn_model(seqVec, targetVec=None, seqLen=None, rnnType="lstm", hiddenNums=256, layerNums=2, batchSize=32, vocaSize=5000):
    with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
        embedMat = tf.get_variable("embedding", [vocaSize, hiddenNums],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
        if rnnType == "rnn":
            cell = tf.nn.rnn_cell.BasicRNNCell(hiddenNums)
        elif rnnType == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenNums)
        else:
            cell = tf.nn.rnn_cell.GRUCell(hiddenNums)

        cells = tf.nn.rnn_cell.MultiRNNCell([cell] * layerNums)
        init_states = cells.zero_state(batchSize, tf.float32)
        if targetVec != None:
            seqVec = tf.nn.embedding_lookup(embedMat, seqVec)
            outputs, _ = tf.nn.dynamic_rnn(cells, seqVec, seqLen, init_states)
            outputs = tf.reshape(outputs, [-1, hiddenNums])
            logits = MLP("Logits", outputs, hiddenNums, vocaSize)
            probs = tf.nn.softmax(logits)
            labels = tf.one_hot(targetVec, vocaSize)
            labels = tf.reshape(labels, [-1, vocaSize])
            loss = tf.reduce_mean(-tf.log(tf.reduce_sum(probs * labels, axis=1) + 1e-10))
            return loss
        else:
            seqVec = tf.nn.embedding_lookup(embedMat, seqVec)
            init_state = cells.zero_state(batchSize, tf.float32)
            output, state = tf.nn.dynamic_rnn(cells, seqVec, initial_state=init_state)
            logit = MLP("Logits", output, hiddenNums, vocaSize)
            prob = tf.nn.softmax(logit)
            predict = tf.argmax(prob, axis=1)
            return predict, state, init_state





