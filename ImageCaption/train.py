import tensorflow as tf
from utils import *
import time
from PIL import Image
from rnn_model import rnn_decoder

import queue
import threading


def img2text(voc, IMG, sess, seqVec, img_input, init_state, img_state, state, wordVal):
    STATE= sess.run([img_state], feed_dict={img_input: IMG[np.newaxis, :, :, :3]})
    WORD = voc.word2id['<start>']
    [WORD, STATE] = sess.run([wordVal, state], feed_dict={seqVec: np.reshape(WORD, [1, 1]), init_state: STATE})
    sent = []
    count = 0
    while voc.id2word[int(WORD)] != '<end>' and count < 50:
        sent.append(int(WORD))
        WORD, STATE = sess.run([wordVal, state], feed_dict={seqVec: np.reshape(WORD, [1, 1]), init_state: STATE})
        count += 1
    return sent

def loop_read():
    while True:
        imgBatch, imgCaption, seq_nums = get_imgBatch_captions_wordLevel(voc, BATCH_SIZE, SEQ_SIZE)
        item = [imgBatch, imgCaption, seq_nums]
        q.put(item)

HIDDEN_NUMS = 512
EMBED_NUMS = 512
LAYER_NUMS = 1
BATCH_SIZE = 32
WORD_NUMS = 8000
SEQ_SIZE = 50
voc = Vocabulary(WORD_NUMS, 'F://BaiduNetdiskDownload//flickr 30k//flickr30k//results_20130124.token')
voc.load_dict()
#Use queue and thread to process data. (update time(1.1s) + read time(0.5s) = 1.6s -> update time(1.1s) + read time(0.005s) = 1.1s)
q = queue.Queue(maxsize=10)
t = threading.Thread(target=loop_read)

def train():
    seqVec = tf.placeholder(tf.int32, [None, None])
    seqLen = tf.placeholder(tf.int32, [None])
    targetVec = tf.placeholder(tf.int32, [None, None])
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    captionNet = rnn_decoder("RNN", LAYER_NUMS, HIDDEN_NUMS, EMBED_NUMS, WORD_NUMS)
    #Training
    probs = captionNet(imgs, seqVec, seqLen, BATCH_SIZE)
    labels = tf.one_hot(targetVec, WORD_NUMS)
    labels = tf.reshape(labels, [-1, WORD_NUMS])
    loss = tf.reduce_mean(-tf.log(tf.reduce_sum(probs * labels, axis=1) + 1e-10))
    #Testing
    probs, wordVal, state, img_state, init_state = captionNet(imgs, seqVec, seqLen, 1)
    Opt = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RNN"))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV2"))
    saver.restore(sess, "./inception_v2_model/inception_v2.ckpt")
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RNN"))
    # saver.restore(sess, "./save_para/.\\model.ckpt")
    t.start()#start thread
    for i in range(200000):
        s = time.time()
        imgBatch, imgCaption, seq_nums = q.get()
        e = time.time()
        read_time = e - s
        s = time.time()
        sess.run(Opt, feed_dict={seqVec: imgCaption[:, :-1], targetVec: imgCaption[:, 1:], imgs: imgBatch, seqLen: seq_nums-1})
        e = time.time()
        update_time = e - s
        if i % 10 == 0:
            LOSS = sess.run(loss, feed_dict={seqVec: imgCaption[:, :-1], targetVec: imgCaption[:, 1:], imgs: imgBatch, seqLen: seq_nums-1})
            print("Iteration: %d, Loss: %f, Read_time: %f, Update_time: %f"%(i, LOSS, read_time, update_time))
        if i % 500 == 0:
            saver.save(sess, "./save_para/model.ckpt")
            IMG = np.array(Image.open('F://BaiduNetdiskDownload//flickr 30k//flickr30k-images//flickr30k-images//205842.jpg').resize([224, 224]))
            caption = img2text(voc, IMG, sess, seqVec, imgs, init_state, img_state, state, wordVal)
            print(voc.vector2sentence(caption))
        pass


if __name__ == "__main__":
    train()