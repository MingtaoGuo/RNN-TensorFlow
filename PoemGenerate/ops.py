import tensorflow as tf
import numpy as np

def MLP(name, inputs, nums_in, nums_out):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.matmul(inputs, W) + b
    return inputs

def generate(sess, char, init_state, state, charVal, char2id, first_char=None):
    sent = []
    if first_char is None:
        CHAR_ID = np.random.randint(0, 7000)
    else:
        CHAR_ID = int(char2id[first_char])
    sent.append(CHAR_ID)
    [CHAR_ID, STATE] = sess.run([charVal, state], feed_dict={char: np.reshape(CHAR_ID, [1, 1])})
    sent.append(int(CHAR_ID))
    count = 0
    while CHAR_ID != char2id['<EOS>'] and count < 100:
        [CHAR_ID, STATE] = sess.run([charVal, state], feed_dict={char: np.reshape(CHAR_ID, [1, 1]), init_state: STATE})
        sent.append(int(CHAR_ID))
        count += 1
    return sent

def BeamSearch(sess, seq_ph, state_ph, state_seq, probs_seq, word2id, beam_size=10):
    CHAR_ID = np.random.randint(0, 7000)
    [PROBS, TOP_K_WORD_VAL, STATE_SEQ] = sess.run([tf.nn.top_k(probs_seq, beam_size)[0], tf.nn.top_k(probs_seq, beam_size)[1], state_seq],
                                                  feed_dict={seq_ph: np.reshape(CHAR_ID, [1, 1])})
    WORD_PROB_SET = []
    for i in range(beam_size):
        WORD_PROB_SET.append([[TOP_K_WORD_VAL[0, i]], np.log(PROBS[0, i] + 1e-10), 0])
    count = 0
    count_break = 0
    while count < beam_size and count_break < 200:
        for i in range(beam_size):
            wordVal = WORD_PROB_SET[i][0][-1]
            [PROBS, TOP_K_WORD_VAL, STATE_SEQ] = sess.run([tf.nn.top_k(probs_seq, beam_size)[0], tf.nn.top_k(probs_seq, beam_size)[1], state_seq],
                                                  feed_dict={seq_ph: np.reshape(wordVal, [1, 1]), state_ph: STATE_SEQ})
            if int(TOP_K_WORD_VAL[0, i]) == int(word2id['<EOS>']):
                count += 1
                continue
            else:
                WORD_PROB_SET[i][0].append(TOP_K_WORD_VAL[0, i])
                WORD_PROB_SET[i][1] += np.log(PROBS[0, i] + 1e-10)
                WORD_PROB_SET[i][2] += 1
        count_break += 1

    max_prob = -np.inf
    max_vec = []
    for i in range(beam_size):
        prob = WORD_PROB_SET[i][1] / WORD_PROB_SET[i][2]
        if prob > max_prob:
            max_prob = prob
            max_vec = WORD_PROB_SET[i][0]
    max_vec.insert(0, CHAR_ID)
    return max_vec