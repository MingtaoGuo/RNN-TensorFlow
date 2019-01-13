from rnn_model import rnn_encoder, rnn_decoder
import numpy as np
import tensorflow as tf
from utils import Vocabulary
import os

# name, layerSize, hiddenSize, embedSize, sourceVocSize, targetVocSize, rnnType='lstm'
BATCH_SIZE = 32
LAYER_SIZE = 1
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


def Eng2Chn(sentence, sess, sourceSeqVec, sourceSeqNums, targetSeqVec, targetSeqNums, probs, states, states_en, init_state):
    seqVec, seqNums = voc.sentence2vector(sentence, flag=0)
    STATE = sess.run([states_en], feed_dict={sourceSeqVec:seqVec, sourceSeqNums: np.reshape(seqNums, [1])})
    WORD_VAL, STATE = sess.run([tf.argmax(probs, axis=1), states], feed_dict={targetSeqVec: np.reshape(CN_WORD2ID['<START>'], [1, 1]), targetSeqNums: np.array([1]), init_state: STATE})
    vector = []
    vector.append(int(WORD_VAL))
    count = 0
    while CN_ID2WORD[int(WORD_VAL)] != '<EOS>' and count < 15:
        WORD_VAL, STATE = sess.run([tf.argmax(probs, axis=1), states], feed_dict={targetSeqVec: np.reshape(WORD_VAL, [1, 1]), targetSeqNums: np.array([1]), init_state: STATE})
        vector.append(int(WORD_VAL))
        count += 1
    print(sentence)
    print(voc.vector2sentence(np.array(vector[:-1]), flag=1))

def translation(sentence):
    sourceSeqVec = tf.placeholder(tf.int32, [None, None])#English
    targetSeqVec = tf.placeholder(tf.int32, [None, None])#Chinese
    sourceSeqNums = tf.placeholder(tf.int32, [None])
    targetSeqNums = tf.placeholder(tf.int32, [None])
    encoder = rnn_encoder("encoder", LAYER_SIZE, HIDDEN_SIZE, EMBED_SIZE, SOURCE_VOC_SIZE, TARGET_VOC_SIZE, RNN_TYPE)
    decoder = rnn_decoder("decoder", LAYER_SIZE, HIDDEN_SIZE, EMBED_SIZE, TARGET_VOC_SIZE, RNN_TYPE)
    states_en = encoder(sourceSeqVec, sourceSeqNums, 1)
    probs, states, init_state = decoder(targetSeqVec, targetSeqNums, 1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\model.ckpt")
    Eng2Chn(sentence, sess, sourceSeqVec, sourceSeqNums, targetSeqVec, targetSeqNums, probs, states, states_en, init_state)

if __name__ == "__main__":
    sentence = "I want to go to school."
    translation(sentence)
