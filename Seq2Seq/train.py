from rnn_model import rnn_encoder, rnn_decoder
import numpy as np
import tensorflow as tf
from utils import Vocabulary
import os

# name, layerSize, hiddenSize, embedSize, sourceVocSize, targetVocSize, rnnType='lstm'
BATCH_SIZE = 32
LAYER_SIZE = 2
HIDDEN_SIZE = 512
EMBED_SIZE = 512
SOURCE_VOC_SIZE = 6000
TARGET_VOC_SIZE = 3000
RNN_TYPE = 'lstm'
LEARNING_RATE = 2e-3
voc = Vocabulary("./cmn.txt", EN_vocSize=SOURCE_VOC_SIZE, CN_vocSize=TARGET_VOC_SIZE)
voc.load_data()
EN_dataset = voc.EN_dataset
EN_seqnums = voc.EN_seqnums
CN_dataset = voc.CN_dataset
CN_seqnums = voc.CN_seqnums
CN_ID2WORD = voc.CN_id2word
CN_WORD2ID = voc.CN_word2id
dataset_nums = int(EN_dataset.shape[0])


def Eng2Chn(sess, sourceSeqVec, sourceSeqNums, targetSeqVec, targetSeqNums, probs, states, states_en, init_state):
    STATE = sess.run([states_en], feed_dict={sourceSeqVec:EN_dataset[5000:5001, :], sourceSeqNums: np.reshape(EN_seqnums[5000], [1])})
    WORD_VAL, STATE = sess.run([tf.argmax(probs, axis=1), states], feed_dict={targetSeqVec: np.reshape(CN_WORD2ID['<START>'], [1, 1]), targetSeqNums: np.array([1]), init_state: STATE})
    vector = []
    vector.append(int(WORD_VAL))
    count = 0
    while CN_ID2WORD[int(WORD_VAL)] != '<EOS>' and count < 15:
        WORD_VAL, STATE = sess.run([tf.argmax(probs, axis=1), states], feed_dict={targetSeqVec: np.reshape(WORD_VAL, [1, 1]), targetSeqNums: np.array([1]), init_state: STATE})
        vector.append(int(WORD_VAL))
        count += 1
    print(voc.vector2sentence(np.array(EN_dataset[5000, 1:-1]), flag=0))
    print(voc.vector2sentence(np.array(vector[:-1]), flag=1))

def train():
    sourceSeqVec = tf.placeholder(tf.int32, [None, None])#English
    targetSeqVec = tf.placeholder(tf.int32, [None, None])#Chinese
    sourceSeqNums = tf.placeholder(tf.int32, [None])
    targetSeqNums = tf.placeholder(tf.int32, [None])
    targetVecLabel = tf.placeholder(tf.int32, [None, None])
    encoder = rnn_encoder("encoder", LAYER_SIZE, HIDDEN_SIZE, EMBED_SIZE, SOURCE_VOC_SIZE, TARGET_VOC_SIZE, RNN_TYPE)
    decoder = rnn_decoder("decoder", LAYER_SIZE, HIDDEN_SIZE, EMBED_SIZE, TARGET_VOC_SIZE, RNN_TYPE)
    #Training phase
    states = encoder(sourceSeqVec, sourceSeqNums, BATCH_SIZE)
    probs, _, _ = decoder(targetSeqVec, targetSeqNums, BATCH_SIZE, states)
    labels = tf.one_hot(targetVecLabel, TARGET_VOC_SIZE)
    labels = tf.reshape(labels, [-1, TARGET_VOC_SIZE])
    loss = tf.reduce_mean(-tf.log(tf.reduce_sum(probs * labels, axis=1) + 1e-10))
    #Test phase
    states_en = encoder(sourceSeqVec, sourceSeqNums, 1)
    probs, states, init_state = decoder(targetSeqVec, targetSeqNums, 1)
    wordVal = tf.argmax(probs, axis=1)
    #Optimization
    Opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, "./save_para/.\\model.ckpt")
    if not os.path.exists("./save_para"):
        os.mkdir("./save_para")

    for i in range(1000000):
        rand_select = np.random.randint(0, dataset_nums, [BATCH_SIZE])
        EN_batch = EN_dataset[rand_select]
        EN_batch_seqnums = EN_seqnums[rand_select]
        CN_batch = CN_dataset[rand_select]
        CN_batch_seqnums = CN_seqnums[rand_select]
        sess.run(Opt, feed_dict={sourceSeqVec: EN_batch, sourceSeqNums: EN_batch_seqnums, targetSeqVec: CN_batch[:, :-1], targetSeqNums: CN_batch_seqnums-1, targetVecLabel: CN_batch[:, 1:]})
        if i % 10 == 0:
            LOSS = sess.run(loss, feed_dict={sourceSeqVec: EN_batch, sourceSeqNums: EN_batch_seqnums, targetSeqVec: CN_batch[:, :-1], targetSeqNums: CN_batch_seqnums-1, targetVecLabel: CN_batch[:, 1:]})
            print("Iteration: %d, Loss: %f"%(i, LOSS))
        if i % 500 == 0:
            saver.save(sess, "./save_para/model.ckpt")
            Eng2Chn(sess, sourceSeqVec, sourceSeqNums, targetSeqVec, targetSeqNums, probs, states, states_en, init_state)


if __name__ == "__main__":
    train()
