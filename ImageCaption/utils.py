import numpy as np
import pandas as pd
from PIL import Image
import time
import nltk



def sentence2vector(sent, char2id, vecLen):
    char_set = set(char2id)
    nums = char_set.__len__()
    idx_start = nums #Start token
    idx_end = nums + 1 # End token
    sent_list = list(sent)
    sentVec = np.ones([vecLen], dtype=np.int32) * char2id[' ']
    sentVec[0] = idx_start
    sentVec[sent_list.__len__() + 1] = idx_end
    for idx, char in enumerate(sent_list):
        if sent_list[idx] in char_set:
            sentVec[idx + 1] = char2id[sent_list[idx]]
        else:
            sentVec[idx + 1] = char2id[' ']
    return sentVec

def vector2sentence1(vector, id2char):
    text = ''
    for i in range(vector.shape[0]):
        text += id2char[vector[i]]
    return text

def saveDict(dictData, path):
    f = open(path, 'w')
    f.write(str(dictData))
    f.close()

def loadDict(path):
    f = open(path, 'r')
    data = f.read()
    return eval(data)

def get_imgBatch_captions(char2id, batchSize = 10, vecLen=100, imgPath='./flickr30k/imgs/', annoPath='./flickr30k/results_20130124.token'):
    while True:
        df = pd.read_table(annoPath)
        chunk = np.array(df.sample(batchSize))
        imgBatch = np.zeros([batchSize, 224, 224, 3])
        captionBatch = np.zeros([batchSize, vecLen])
        for i in range(batchSize):
            imgBatch[i, :, :, :] = np.array(Image.open(imgPath+chunk[i, 0][:-2]).resize([224, 224]))[:, :, :3]
            captionBatch[i, :] = sentence2vector(chunk[i, 1], char2id, vecLen)
        yield (imgBatch, captionBatch)

class Vocabulary:
    def __init__(self, vocSize, filepath):
        self.vocSize = vocSize
        self.word2id = {}
        self.id2word = {}
        self.word_count = {}
        self.filepath = filepath

    def build_dict(self):
        #Generate vocabulary dictionary: word2id, id2word
        captions = np.array(pd.read_table(self.filepath))[:, 1]
        for i in range(captions.shape[0]):
            for word in nltk.word_tokenize(captions[i].lower()):
                if word not in self.word_count.keys():
                    self.word_count[word] = 1
                else:
                    self.word_count[word] += 1
        word_count = sorted(list(self.word_count.items()), key=lambda x:x[1], reverse=True)
        for i in range(self.vocSize-3):
            self.word2id[word_count[i][0]] = i
            self.id2word[i] = word_count[i][0]
        self.word2id['<end>'] = self.vocSize - 1
        self.id2word[self.vocSize - 1] = '<end>'
        self.word2id['<unk>'] = self.vocSize - 2
        self.id2word[self.vocSize - 2] = '<unk>'
        self.word2id['<start>'] = self.vocSize - 3
        self.id2word[self.vocSize - 3] = '<start>'
        saveDict(self.word2id, "./voc_dict/word2id.txt")
        saveDict(self.id2word, "./voc_dict/id2word.txt")

    def load_dict(self):
        self.word2id = loadDict("./voc_dict/word2id.txt")
        self.id2word = loadDict("./voc_dict/id2word.txt")

    def sentence2vector(self, sent: str, vecSize: int):
        sent = nltk.word_tokenize(sent.lower())
        sentVec = np.zeros([1, vecSize])
        allWord = set(self.word2id)
        sentVec[0, 0] = self.word2id['<start>']
        for idx, word in enumerate(sent):
            if idx >= vecSize - 3:
                break
            if word in allWord:
                sentVec[0, idx + 1] = self.word2id[word]
            else:
                sentVec[0, idx + 1] = self.word2id['<unk>']
        sentVec[0, idx + 2] = self.word2id['<end>']
        seqNums = idx + 3
        return sentVec, seqNums

    def vector2sentence(self, vector):
        sent = ''
        for i in range(vector.__len__()):
            sent += self.id2word[vector[i]] + ' '
        return sent

def vector2sentence(vector, id2word):
    sent = ''
    for i in range(vector.shape[0]):
        sent += id2word[vector[i]] + ' '
    return sent

def get_imgBatch_captions_wordLevel(voc, batchSize = 16, vecLen=50, imgPath='./flickr30k/imgs/', annoPath='./flickr30k/results_20130124.token'):
    df = pd.read_table(annoPath)
    chunk = np.array(df.sample(batchSize))
    imgBatch = np.zeros([batchSize, 224, 224, 3])
    captionBatch = np.zeros([batchSize, vecLen])
    seqNums = np.zeros([batchSize], dtype=np.int32)
    for i in range(batchSize):
        imgBatch[i, :, :, :] = np.array(Image.open(imgPath+chunk[i, 0][:-2]).resize([224, 224]))[:, :, :3]
        captionBatch[i, :], seqNums[i] = voc.sentence2vector(chunk[i, 1], vecLen)
    return imgBatch, captionBatch, seqNums

# if __name__ == "__main__":
#     voc = Vocabulary(8000, 'F://BaiduNetdiskDownload//flickr 30k//flickr30k//results_20130124.token')
#     voc.build_dict()
#     # voc.load_dict()
#     a, b = voc.sentence2vector("a man in black shirt is running", 50)
#     _, sents, seqNums = get_imgBatch_captions_wordLevel(voc)
