import numpy as np
import os



class poemsTxt:
    def __init__(self, path):
        self.char2id = {}
        self.id2char = {}
        self.path = path
        self.trainingData = 0
        self.trainingDataSeqLens = 0

    def build(self):
        filenames = os.listdir(self.path)
        dataset = []
        text = ''
        for filename in filenames:
            with open(self.path+filename, 'r') as f:
                oneTxt = f.read().strip().replace('\n', '')
                if oneTxt == '':
                    continue
                else:
                    onePoem = list(oneTxt.replace('\n', ''))
                    if '《' in onePoem or '》' in onePoem or '_' in onePoem or ':' in onePoem or '（' in onePoem or '）' in onePoem or '“' in onePoem or '”' in onePoem:
                        continue
                    dataset.append(onePoem)
                    text += oneTxt
                f.close()
        allChar = list(set(text))
        charNums = allChar.__len__()
        for i in range(charNums):
            self.char2id[allChar[i]] = i
            self.id2char[i] = allChar[i]
        self.char2id['<EOS>'] = charNums
        self.id2char[charNums] = '<EOS>'

        nums = dataset.__len__()
        datasetNums = 0
        for i in range(nums):
            seqNums = dataset[i].__len__()
            if seqNums < 50 and seqNums >= 10:
                datasetNums += 1
        self.trainingData = np.zeros([datasetNums, 50], dtype=np.int32)
        self.trainingDataSeqLens = np.zeros([datasetNums], dtype=np.int32)
        count = 0
        for i in range(nums):
            seq = dataset[i]
            seqLen = seq.__len__()
            if seqLen < 50 and seqLen >= 10:
                for j in range(seqLen):
                    self.trainingData[count, j] = self.char2id[seq[j]]
                self.trainingData[count, seqLen] = self.char2id['<EOS>']
                self.trainingDataSeqLens[count] = seqLen + 1
                count += 1
        if not os.path.exists('./poemData'):
            os.mkdir('./poemData')
        np.savetxt("./poemData/trainingData.txt", self.trainingData, fmt="%d")
        np.savetxt("./poemData/trainingDataSeqLen.txt", self.trainingDataSeqLens, fmt="%d")
        saveDict("./poemData/char2id.txt", self.char2id)
        saveDict("./poemData/id2char.txt", self.id2char)

    def load_data(self):
        self.char2id = loadDict("./poemData/char2id.txt")
        self.id2char = loadDict("./poemData/id2char.txt")
        trainingData = np.loadtxt("./poemData/trainingData.txt")
        trainingDataSeqLen = np.loadtxt("./poemData/trainingDataSeqLen.txt")
        return trainingData, trainingDataSeqLen

    def vector2sentence(self, vector):
        sent = ''
        for i in range(vector.__len__()):
            char = vector[i]
            if char == self.char2id["<EOS>"]:
                break
            sent += self.id2char[char]
        return sent

def saveDict(path, dictData):
    f = open(path, 'w')
    f.write(str(dictData))
    f.close()

def loadDict(path):
    f = open(path, 'r')
    data = eval(f.read())
    f.close()
    return data






