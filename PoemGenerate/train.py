import tensorflow as tf
from utils import poemsTxt
from ops import generate
import numpy as np
from model import rnn_model

HIDDEN_NUMS = 256
LAYER_NUMS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
MAX_ITERATION = 20000


def train():
    poem = poemsTxt("./tangshiDataset/dataset/")
    trainData, trainSeqLen = poem.load_data()
    char2id = poem.char2id
    charNums = char2id.__len__()
    seqVec = tf.placeholder(tf.int32, [None, None])
    seqLen = tf.placeholder(tf.int32, [None])
    seqTarget = tf.placeholder(tf.int32, [None, None])
    #training
    loss = rnn_model(seqVec, seqTarget, seqLen, rnnType="lstm", hiddenNums=HIDDEN_NUMS, layerNums=LAYER_NUMS, batchSize=BATCH_SIZE, vocaSize=charNums)
    #testing
    pred, state, init_state = rnn_model(seqVec, rnnType="lstm", hiddenNums=HIDDEN_NUMS, layerNums=LAYER_NUMS, batchSize=1, vocaSize=charNums)
    Opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    nums = int(trainData.shape[0])
    for i in range(MAX_ITERATION):
        randSample = np.random.randint(0, nums, [BATCH_SIZE])
        batch = trainData[randSample, :-1]
        batchTarget = trainData[randSample, 1:]
        batchLen = trainSeqLen[randSample] - 1
        sess.run(Opt, feed_dict={seqVec: batch, seqTarget: batchTarget, seqLen: batchLen})
        if i % 10 == 0:
            LOSS = sess.run(loss, feed_dict={seqVec: batch, seqTarget: batchTarget, seqLen: batchLen})
            print("Iteration: %d, Loss: %f"%(i, LOSS))
        if i % 500 == 0:
            vector = generate(sess, seqVec, init_state, state, pred, char2id)
            print(poem.vector2sentence(vector))
            saver.save(sess, "./save_para/model.ckpt")

if __name__ == "__main__":
    train()

