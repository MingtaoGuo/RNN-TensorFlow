import numpy as np
from nltk.tokenize import word_tokenize
import os

class Vocabulary:
    def __init__(self, path, EN_vocSize, CN_vocSize):
        self.EN_word2id = {}
        self.EN_id2word = {}
        self.CN_word2id = {}
        self.CN_id2word = {}
        self.EN_dataset = []
        self.CN_dataset = []
        self.EN_seqnums = 0
        self.CN_seqnums = 0
        self.path = path
        self.EN_vocSize = EN_vocSize
        self.CN_vocSize = CN_vocSize

    def build_dict(self):
        #Calculate the number of dataset
        with open(self.path, 'r', encoding="utf-8") as f:
            line = '.'
            nums_dataset = -1
            while line != '':
                nums_dataset += 1
                line = f.readline()
            f.close()
        #Build dataset
        with open(self.path, 'r', encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                pair = line.strip().split('\t')
                EN_sentence = word_tokenize(pair[0].lower())
                CN_sentence = list(pair[1])
                self.EN_dataset.append(EN_sentence)
                self.CN_dataset.append(CN_sentence)
            f.close()
        #Build dictionary
        EN_word_count = {}
        CN_word_count = {}
        for i in range(nums_dataset):
            for word in self.EN_dataset[i]:
                if word not in list(EN_word_count.keys()):
                    EN_word_count[word] = 1
                else:
                    EN_word_count[word] += 1
            for word in self.CN_dataset[i]:
                if word not in list(CN_word_count.keys()):
                    CN_word_count[word] = 1
                else:
                    CN_word_count[word] += 1
        EN_word_count = sorted(list(EN_word_count.items()), key=lambda x:x[1], reverse=True)[:self.EN_vocSize]
        CN_word_count = sorted(list(CN_word_count.items()), key=lambda x:x[1], reverse=True)[:self.CN_vocSize]
        for i in range(self.EN_vocSize - 3):
            word = EN_word_count[i][0]
            self.EN_word2id[word] = i
            self.EN_id2word[i] = word
        self.EN_word2id["<EOS>"], self.EN_id2word[i + 1] = i + 1, "<EOS>"
        self.EN_word2id["<UNK>"], self.EN_id2word[i + 2] = i + 2, "<UNK>"
        self.EN_word2id["<START>"], self.EN_id2word[i + 3] = i + 3, "<START>"
        for i in range(self.CN_vocSize - 3):
            word = CN_word_count[i][0]
            self.CN_word2id[word] = i
            self.CN_id2word[i] = word
        self.CN_word2id["<EOS>"], self.CN_id2word[i + 1] = i + 1, "<EOS>"
        self.CN_word2id["<UNK>"], self.CN_id2word[i + 2] = i + 2, "<UNK>"
        self.CN_word2id["<START>"], self.CN_id2word[i + 3] = i + 3, "<START>"
        #Dataset(sentences) to training dataset(vector)
        count = 0
        for i in range(nums_dataset):
            if self.EN_dataset[i].__len__() < 15 and self.CN_dataset[i].__len__() < 15:
                count += 1
        EN_dataset = np.zeros([count, 17], dtype=np.int32)
        EN_seqnums = np.zeros([count], dtype=np.int32)
        CN_dataset = np.zeros([count, 17], dtype=np.int32)
        CN_seqnums = np.zeros([count], dtype=np.int32)
        count = 0
        allEN = set(self.EN_word2id)
        allCN = set(self.CN_word2id)
        for i in range(nums_dataset):
            if self.EN_dataset[i].__len__() < 15 and self.CN_dataset[i].__len__() < 15:
                #Process English word
                seqnums = self.EN_dataset[i].__len__()
                EN_dataset[count, 0] = self.EN_word2id["<START>"]
                for j in range(seqnums):
                    if self.EN_dataset[i][j] in allEN:
                        EN_dataset[count, j+1] = self.EN_word2id[self.EN_dataset[i][j]]
                    else:
                        EN_dataset[count, j+1] = self.EN_word2id["<UNK>"]
                EN_dataset[count, j+2] = self.EN_word2id["<EOS>"]
                EN_seqnums[count] = seqnums + 2
                #Process Chinese word
                seqnums = self.CN_dataset[i].__len__()
                CN_dataset[count, 0] = self.CN_word2id["<START>"]
                for j in range(seqnums):
                    if self.CN_dataset[i][j] in allCN:
                        CN_dataset[count, j+1] = self.CN_word2id[self.CN_dataset[i][j]]
                    else:
                        CN_dataset[count, j+1] = self.CN_word2id["<UNK>"]
                CN_dataset[count, j+2] = self.CN_word2id["<EOS>"]
                CN_seqnums[count] = seqnums + 2
                count += 1
        path = "./dataset_voc"
        if not os.path.exists(path):
            os.mkdir(path)
        np.savetxt(path+"/"+"EN_dataset.txt", EN_dataset)
        np.savetxt(path+"/"+"EN_seqnums.txt", EN_seqnums)
        np.savetxt(path+"/"+"CN_dataset.txt", CN_dataset)
        np.savetxt(path+"/"+"CN_seqnums.txt", CN_seqnums)
        self.save_dict()

    def save_dict(self):
        path = "./dataset_voc"
        with open(path + "/" + "ENword2id.txt", 'w', encoding="utf-8") as f:
            f.write(str(self.EN_word2id))
            f.close()
        with open(path + "/" + "ENid2word.txt", 'w', encoding="utf-8") as f:
            f.write(str(self.EN_id2word))
            f.close()
        with open(path + "/" + "CNword2id.txt", 'w', encoding="utf-8") as f:
            f.write(str(self.CN_word2id))
            f.close()
        with open(path + "/" + "CNid2word.txt", 'w', encoding="utf-8") as f:
            f.write(str(self.CN_id2word))
            f.close()


    def load_data(self):
        path = "./dataset_voc"
        if not os.path.exists(path):
            self.build_dict()
            self.EN_dataset = np.loadtxt("./dataset_voc/EN_dataset.txt")
            self.EN_seqnums = np.loadtxt("./dataset_voc/EN_seqnums.txt")
            self.CN_dataset = np.loadtxt("./dataset_voc/CN_dataset.txt")
            self.CN_seqnums = np.loadtxt("./dataset_voc/CN_seqnums.txt")
        else:
            with open(path + "/" + "ENword2id.txt", 'r', encoding="utf-8") as f:
                self.EN_word2id = eval(f.read())
                f.close()
            with open(path + "/" + "ENid2word.txt", 'r', encoding="utf-8") as f:
                self.EN_id2word = eval(f.read())
                f.close()
            with open(path + "/" + "CNword2id.txt", 'r', encoding="utf-8") as f:
                self.CN_word2id = eval(f.read())
                f.close()
            with open(path + "/" + "CNid2word.txt", 'r', encoding="utf-8") as f:
                self.CN_id2word = eval(f.read())
                f.close()
            self.EN_dataset = np.loadtxt("./dataset_voc/EN_dataset.txt")
            self.EN_seqnums = np.loadtxt("./dataset_voc/EN_seqnums.txt")
            self.CN_dataset = np.loadtxt("./dataset_voc/CN_dataset.txt")
            self.CN_seqnums = np.loadtxt("./dataset_voc/CN_seqnums.txt")

    def vector2sentence(self, vector, flag=0):
        #flag=0: EN, flag=1: CN
        sent = ''
        if flag == 0:
            for i in range(vector.shape[0]):
                sent += self.EN_id2word[vector[i]] + ' '
        else :
            for i in range(vector.shape[0]):
                sent += self.CN_id2word[vector[i]]
        return sent

    def sentence2vector(self, sent, flag=0):
        #flag=0: EN, flag=1: CN
        vector = np.zeros([1, 17])
        seqNums = 0
        if flag == 0:
            vector[0, 0] = self.EN_word2id['<START>']
            list_sent = word_tokenize(sent.lower())
            for idx, word in enumerate(list_sent):
                vector[0, idx + 1] = self.EN_word2id[word]
            vector[0, idx + 2] = self.EN_word2id['<EOS>']
            seqNums = list_sent.__len__() + 2
        else:
            list_sent = list(sent)
            vector[0, 0] = self.CN_word2id['<START>']
            for idx, word in enumerate(list_sent):
                vector[0, idx + 1] = self.CN_word2id[word]
            vector[0, idx + 2] = self.EN_word2id['<EOS>']
            seqNums = list_sent.__len__() + 2
        return vector, seqNums







if __name__ == "__main__":
    voc = Vocabulary("cmn.txt", 6000, 3000)
    voc.load_data()
    voc.vector2sentence()