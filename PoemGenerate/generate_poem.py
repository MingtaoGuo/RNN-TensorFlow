import tensorflow as tf
from model import rnn_model
from utils import poemsTxt
from ops import generate

HIDDEN_NUMS = 256
LAYER_NUMS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
MAX_ITERATION = 10000

poem = poemsTxt("")
_, _, = poem.load_data()
charNums = poem.char2id.__len__()

def generatePoem(first_char):
    seqVec = tf.placeholder(tf.int32, [None, None])
    # testing
    pred, state, init_state = rnn_model(seqVec, rnnType="lstm", hiddenNums=HIDDEN_NUMS, layerNums=LAYER_NUMS,
                                        batchSize=1, vocaSize=charNums)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\model.ckpt")
    onePoem = generate(sess, seqVec, init_state, state, pred, poem.char2id, first_char)

    print(poem.vector2sentence(onePoem))

if __name__ == "__main__":
    generatePoem("龙")
