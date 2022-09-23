import numpy as np
import json
## import requests
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

DIST = 10
COS = 0.99

class Preprocess:
    def __init__(self, position = ["dr", "ha", "hu", "it", "ku", "gn"]):
        self.inks = []
        self.words = []
        self.urls = []
        self.uid = []
        self.position = position
        self.dataset = []

    # クラス内の変数にデータを格納
    def importJson(self,lan):
        print("importing data...")
        a = open('./data/export.json')
        b = json.load(a)
        samples = b["medical-samples"]

        # 全単語取得
        if lan == "all":
            for x in range(len(samples)):
                if not samples[x] is None:
                    # lower() →　全て小文字にする
                    uid = samples[x]["userId"].lower()
                    for p in self.position:
                        # startswith: 文字列の最初が一致するか判定する
                        if uid.startswith('user_' + p.lower()):
                            self.inks.append(samples[x]["data"]["request"]["inks"])
                            self.words.append(samples[x]["word"])
                            self.urls.append(samples[x]["imageUrl"])
                            self.uid.append(uid)
                            break
        # 英語 or バングラ語のみ
        else:
            for x in range(len(samples)):
                if not samples[x] is None:
                    # lower() →　全て小文字にする
                    uid = samples[x]["userId"].lower()
                    for p in self.position:
                        # startswith: 文字列の最初が一致するか判定する
                        if uid.startswith('user_' + p.lower()):
                            # 言語判定
                            if lan == samples[x]["data"]["request"]["language"]:
                                self.inks.append(samples[x]["data"]["request"]["inks"])
                                self.words.append(samples[x]["word"])
                                self.urls.append(samples[x]["imageUrl"])
                                self.uid.append(uid)
                                break
                            else:
                                pass
        print("done importing data")

    # imgUrlから写真を取ってくる
    def download_img(self, url, file_name):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(r.content)

    # listをバイナリ化してテキストデータで保存
    def saveLists(self, d, name):
        with open("./data/all/" + name + ".txt", "wb") as fp:   #Pickling
            pickle.dump(d, fp)


    # それぞれの変数をtxtファイルに保存
    def preprocess(self, lan):
        # ラベル付をするために使用
        self.saveLists(self.inks, lan + "_inks")
        self.saveLists(self.words,lan + "_words")
        self.saveLists(self.urls, lan + "_urls")
        self.saveLists(self.uid, lan + "_uid")

    # ストローク番号を追加する関数
    def stroke(self):
        for y in range(len(self.inks)):
            for x in range(len(self.inks[y])):
                self.inks[y][x].append([x] * len(self.inks[y][x][0]))

    #　ストロークをつなげて転置
    def strokeConnect(self):
        print("connecting strokes")
        for y in range(len(self.inks)): # 1文字ごとの作業
            f = self.inks[y][0]
            f = np.array(f).T
            for x in range(len(self.inks[y]) - 1): # 1ストロークごとの作業
                e = self.inks[y][x+1]
                e = np.array(e).T
                f = np.append(f, e, axis = 0)
            self.dataset.append(f)
        print("done connecting strokes")

    # 時間情報を削除
    def removeTime(self):
        b = []
        for x in range(len(self.dataset)):
            a = np.array(self.dataset[x]).T
            a = np.delete(a, 2, 0)
            a = a.T
            b.append(a)
        self.dataset = b

    # 近接する点の除去
    def listClosePoint(self):
        print("removing extra points")
        l = []
        for c in self.dataset:
            a = []
            for i in range(len(c)):
                if i == 0 or i+1 == len(c):
                    # 文字の最初のポイントと最後のポイントはそのまま残す
                    a.append(c[i])
                elif c[i-1][2] != c[i][2] or c[i][2] != c[i+1][2]:
                    # ストロークの最初のポイントと最後のポイントはそのまま残す
                    a.append(c[i])
                else:
                    dp = math.sqrt((c[i][0]-c[i-1][0])**2 + (c[i][1]-c[i-1][1])**2)
                    if dp >= DIST:
                        a.append(c[i])
            if not a is None:
                l.append(a)
        self.dataset = l

    # 直線上の点の除去
    def listStraightPoint(self):
        l = []
        for c in self.dataset:
            a = []
            for i in range(len(c)):
                if i == 0 or i+1 == len(c):
                    # 文字の最初のポイントと最後のポイントはそのまま残す
                    a.append(c[i])
                elif c[i-1][2] != c[i][2] or c[i][2] != c[i+1][2]:
                    # ストロークの最初のポイントと最後のポイントはそのまま残す
                    a.append(c[i])
                else:
                    dx0 = c[i][0] - c[i-1][0] # deltaX(i-1)
                    dx1 = c[i+1][0] - c[i][0] # deltaX(i)
                    dy0 = c[i][1] - c[i-1][1] # deltaY(i-1)
                    dy1 = c[i+1][1] - c[i][1] # deltaY(i)
                    if (((dx0**2+dy0**2)**0.5)*((dx1**2+dy1**2)**0.5)) != 0:
                        cp = (dx0*dx1 + dy0*dy1) / (((dx0**2+dy0**2)**0.5)*((dx1**2+dy1**2)**0.5))
                        if cp <= COS:
                            a.append(c[i])
            if not a is None:
                l.append(a)
        self.dataset = l
        print("done removing extra points")

    # 正規化　min-max
    def normalization(self, lan):
        print("normalization")
        a = self.dataset[0]
        for i in range(len(self.dataset) -1):
            a = np.vstack((a, self.dataset[i+1]))

        xma = max(a.T[0])
        xmi = min(a.T[0])
        yma = max(a.T[1])
        ymi = min(a.T[1])
        ndata = []
        for c in self.dataset:
            n1 = (np.array(c).T[0] - xmi) / (xma - xmi)
            n2 = (np.array(c).T[1] - ymi) / (yma - ymi)
            nor_c = [n1, n2, np.array(c).T[2]]
            nor_c = np.array(nor_c).T
            ndata.append(nor_c)
        self.dataset = ndata
        self.saveLists(self.dataset, lan + "_pped_data")
        print("done normalization")

    # 点を線に変換する
    def pointToLine(self):
        print("converting points to lines")
        l = []
        for c in self.dataset:
            a = []
            for i in range(len(c) - 1):
                    x = c[i][0] # 線の始点のx座標
                    y = c[i][1] # 線の始点のy座標
                    dx = c[i+1][0] - c[i][0] # 2点間のx軸の距離
                    dy = c[i+1][1] - c[i][1] # 2点間のy軸の距離
                    sst = int(c[i][2] == c[i+1][2]) # 同じストロークか
                    dst = int(c[i][2] != c[i+1][2]) # 異なるストロークか
                    line = [x, y, dx, dy, sst, dst]
                    a.append(line)
            l.append(a)
        self.dataset = l
        print("done converting points to lines")


    def runThis(self):
        self.importJson()
        self.preprocess()
        self.stroke()
        self.strokeConnect()
        self.removeTime()
        self.listStraightPoint()
        self.listClosePoint()

        # self.normalization()
        # self.pointToLine()
        print("done preprocessing")
# ------------------------------------------------

# lan = ["0","all","en","bn"]
# label = ["0","learning","image-title"]
def preprocess_main(lan):
    pp = Preprocess()
    pp.importJson(lan)
    pp.preprocess(lan)
    pp.stroke()
    pp.strokeConnect()
    pp.removeTime()
    pp.listStraightPoint()
    pp.listClosePoint()
    # 画像用も正規化を行うことに決定
    pp.normalization(lan)
    print("done preprocessing")

# -----------------------------------------------
if __name__ == '__main__':
    print("Do-Nathing")
