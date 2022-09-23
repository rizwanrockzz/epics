import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle

class Label_gen:
    def __init__(self, position = ["dr", "ha", "hu", "it", "ku", "gn"]):
        self.corpus = []
        self.words = []
        self.labels = []
        self.position = position
        self.num_words = 480 # 認識する用語の数

    # 単語データを取ってくる
    def getList(self):
        text_file = open("./data/dataList.txt", "r")
        a = text_file.read().split('\n')
        # 最後の空白も読んじゃうのでスライスで消す
        self.corpus = a[0:self.num_words]
        text_file.close()

    # jsonからとってきた正解データを取ってくる
    def getData(self, lan):
        f = open("./data/all/" + lan + "_words.txt", "rb")
        self.words = pickle.load(f)
        f.close()

    # ラベルデータ作成
    def toBinary(self, lan):
        idx = []
        for w in self.words:
            if w in self.corpus:
                a = self.corpus.index(w)
            else:
                a = None
            idx.append(a)
        labels = []
        for x in idx:
            b = np.zeros((self.num_words,), dtype=int)
            # 実験に含みたくないデータはlabelに-1を入れる
            if x is None:
                b[0] = -1
            else:
                b[x] = 1
            labels.append(b)
        self.labels = labels
        with open("./data/all/" + lan + "_bined_labels.txt", "wb") as fp:   #Pickling
            pickle.dump(self.labels, fp)

    def toImageTitle(self, lan):
        labels = []
        count = 0
        count_none = 0
        for w in self.words:
            if w in self.corpus:
                count += 1
                a = w
            else:
                count_none += 1
                a = None
            labels.append(a)

        print("toImage sample:"+str(count)+" None:"+str(count_none))
        self.labels = labels
        with open("./data/all/" + lan + "_word_labels.txt", "wb") as fp:   #Pickling
            pickle.dump(self.labels, fp)

    def runThis(self):
        self.getList()
        self.getData()
        self.toBinary()
        print("done label generation")
# ----------------------------------------------------------------

# lan = ["0", "all", "en", "bn"]
# label_type = ["0","learning","image-title"]
def labels_main(lan, label_type):
    lg = Label_gen()
    # 正解データ取得（480単語）
    lg.getList()
    lg.getData(lan)
    if label_type == "learning":
        lg.toBinary(lan)
        print("done label generation")
    elif label_type == "image-title":
        lg.toImageTitle(lan)
        print("done label generation")
    else:
        print("Label_type_Error")

# ---------------------------------------------------------------
if __name__ == '__main__':
    print("Do-Nathing")
