import numpy as np
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import pickle
import random
import gc
from prepro import Preprocess

from tqdm import tqdm

class Augment:
    def __init__(self, dataset, num, ratio):
        # print(dataset)
        self.dataset = dataset
        self.original = dataset
        self.label = "Multivit plus"
        self.Rotate_num = num[0]
        self.Parallel_num = num[1]
        self.Ratio_num = num[2]
        self.ratio = ratio
        self.maxlen = 0


# SMOTE用の関数たち
    def padding(self, d):
        xdata = sequence.pad_sequences(d, maxlen=self.maxlen, padding = 'post', dtype=np.float32)
        print("done padding")
        del d
        return xdata

    # def smote(self):
    #     sm = SMOTE(random_state=42)
    #     # l = []
    #     a = np.empty((len(self.dataset), self.maxlen*6))
    #     for i, d in enumerate(self.dataset):
    #         a[i] = np.reshape(d, (self.maxlen*6,))
    #         # l.append(np.reshape(d, (self.maxlen*6,)))
    #     self.dataset = a
    #     self.dataset, self.labels = sm.fit_sample(self.dataset, self.labels)
    #     b = np.empty(len(self.dataset), (self.maxlen, 6))
    #     for i, d in enumerate(self.dataset):
    #         b[i] = np.reshape(d, (self.maxlen, 6))
    #     self.dataset = b
    #     # return X_res, Y_res

#ここまで

    # ストロークを平行移動
    def genParallel(self, data):
        print("data augmentation(parallel)")
        a = np.array(data)
        d = []
        labels = []

        ###### 1ワードごとにストロークの値を0に戻す、ランダムノイズの値を書き直す、lを初期化する
        for i, w in enumerate(a):
            st = 0
            xn = random.gauss(0, 0.010)
            yn = random.gauss(0, 0.010)
            # xn = random.uniform(-per, per)
            # yn = random.uniform(-per, per)
            l = []
            # 1つのストロークには同じノイズを乗せる（平行移動）
            for p in w:
                # 同じストローク
                if p[2] == st:
                    pass
                # ストロークが変わる際にノイズの値を書き直す
                else:
                    xn = random.gauss(0, 0.010)
                    yn = random.gauss(0, 0.010)
                    # xn = random.uniform(-per, per)
                    # yn = random.uniform(-per, per)
                    st += 1

                p0 = p[0] + xn
                p1 = p[1] + yn

                # x座標, y座標, ストローク
                l.append([p0, p1, p[2]])
            d.append(l)

        self.dataset = d
        del a, d, l, xn, yn

    #####  genRotate  --> strokeRotate  --> pointRotate
    # 点の回転
    def pointRotate(self, point, center, angle):
        # x, y = point
        # c_x, c_y = center
        rad = math.radians(angle)
        # 点の回転用の行列
        mat = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        xy = np.array([point[0]-center[0], point[1]-center[1]])
        # 点の回転の式
        XY = np.dot(mat, xy)
        X = XY[0] + center[0]
        Y = XY[1] + center[1]
        return X, Y


    def strokeRotate(self, l, dispersion):
        ien = len(l) - 1
        c_x = (l[0][0]+l[ien][0])/2
        c_y = (l[0][1]+l[ien][1])/2
        newp = []
        randAngle = random.gauss(0, dispersion)
        # randAngle = random.uniform(-angle, angle)
        for p in l:
            x, y = p[0], p[1]
            X, Y = self.pointRotate([x, y], [c_x, c_y], randAngle)
            newp.append([X, Y, p[2]])
        return newp


    def genRotate(self, data, dispersion):
        print("data augmentation(rotate)")
        a = np.array(data)
        d = []
        l = []

        # ワードごとにストロークカウンタを1に
        # ワードごとにcを空に
        # a: [word,,,word]
        for j, w in enumerate(a):
            st = 1
            c = []
            # w: [[x,y,st],,,[x,y,st]]
            # i: index, p: [x,y,st]
            for i, p in enumerate(w):
                # ワードの最後の点なら追加してrotate
                if i+1 == len(w):
                    l.append(p)
                    st += 1
                    newl = self.strokeRotate(l, dispersion)
                    l = []
                    c.extend(newl)
                elif w[i+1][2] == st:
                    l.append(p)
                    st += 1
                    newl = self.strokeRotate(l, dispersion)
                    c.extend(newl)
                    l = []
                # 終点でなければそのまま追加
                elif w[i+1][2] != st:
                    l.append(p)
            d.append(c)

        self.dataset = d
        del a, d, l, c, st


    # 風の比率変換プログラム
    def genRatio(self, data, num, ratio):
        print("data augmentation(XYratio)")
        a = np.array(data)
        d = []
        
        # i : index,  w : word (0(x),1(y),2(stroke))
        for _, w in enumerate(a):
            l = []
            # Mean = [x, y, stroke]
            Mean = np.mean(w, axis=0)

            for p in w:
                # X方向での拡張(1)
                if num == 1:
                    if p[0] >= Mean[0]:
                        p0 = p[0] + p[0]*ratio[0]
                    else: #p[0] < Mean[0]
                        p0 = p[0] - p[0]*ratio[0]

                    p1 = p[1]
                # Y方向での拡張(2)
                else:
                    if p[1] >= Mean[1]:
                        p1 = p[1] + p[1]*ratio[0]
                    else: #p[1] < Mean[1]
                        p1 = p[1] - p[1]*ratio[0]

                    p0 = p[0]

                # x座標, y座標, ストローク
                l.append([p0, p1, p[2]])

            d.append(l)

        self.dataset = d
        del a, l, d, ratio, p0, p1, Mean

    # 画像変換では使用できない
    def pointToLine(self, data):
        print("converting points to lines")
        l = []
        for c in tqdm(data):
            a = []
            for i in range(len(c) - 1):
                x = c[i][0] # 線の始点のx座標
                y = c[i][1] # 線の始点のy座標
                dx = c[i+1][0] - c[i][0] # 2点間のx軸の距離
                dy = c[i+1][1] - c[i][1] # 2点間のy軸の距離
                sst = int(c[i][2] == c[i+1][2]) # 同じストロークか
                dst = int(c[i][2] != c[i+1][2]) # 異なるストロークか
                line = np.array([x, y, dx, dy, sst, dst])
                a.append(line)
            a = np.array(a)
            l.append(a)
        self.dataset = np.array(l)
        print("done converting points to lines")

    # 拡張後の座標データを画像に変換する
    def Chara_img(self, data, augment_type):
        # data = [[word],,,[word]]
        # word = [[x, y, s],,,[x, y, s]]
        # label = [[word_label],,,[word_label]]

        original_word = np.array(self.original[0])
        auged_word = data[len(data)-1] #出力単語の決定
        auged_word = np.array(auged_word)
        st = 0
        d = []
        l = []
        # ストロークごとに線を分ける
        for index, p in enumerate(auged_word):
            if p[2] == st:
                # l = [[x,y,st],,,[x,y,st]]
                l.append(p)
            else:
                # l = [[x,,,x],[y,,,y],[st,,,st]]
                l = np.array(l).T

                d.append(l)
                l = []
                l.append(p)
                st+=1
            if index == len(auged_word)-1:
                l = np.array(l).T
                d.append(l)
                l = []
                l.append(p)
                st+=1

        for l in d:
            plt.plot(l[0], l[1], marker = "o", color="b")

        st = 0
        d = []
        l = []
        # ストロークごとに線を分ける
        for index, p in enumerate(auged_word):
            if p[2] == st:
                # l = [[x,y,st],,,[x,y,st]]
                l.append(p)
            else:
                # l = [[x,,,x],[y,,,y],[st,,,st]]
                l = np.array(l).T

                d.append(l)
                l = []
                l.append(p)
                st+=1
            if index == len(auged_word)-1:
                l = np.array(l).T
                
                d.append(l)
                l = []
                l.append(p)
                st+=1

        for l in d:
            plt.plot(l[0], l[1], marker = "o", color="r")


        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])

        plt.tick_params(
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False,
                right=False,
                top=False
                )

        plt.savefig("./data/"+str(augment_type)+".png",bbox_inches="tight", pad_inches=0.15)
        plt.close()
        print(str(augment_type) + " : create img")

# --------------------------------------------------------------------------------

    def runThis(self, now=None):
        self.Chara_img(self.dataset,augment_type="NoAug:" + str(now))
        if self.Rotate_num > 0:
            self.genRotate(self.dataset,dispersion=2)
            gc.collect()
            self.Chara_img(self.dataset,augment_type="Rotate:" + str(now))
        if self.Parallel_num > 0:
            self.genParallel(self.dataset)
            gc.collect()
            self.Chara_img(self.dataset, augment_type="Parallel:" + str(now))
        if self.Ratio_num > 0:
            self.genRatio(self.dataset, num=self.Ratio_num, ratio=self.ratio)
            gc.collect()
            self.Chara_img(self.dataset,augment_type="Ratio:(" + str(self.ratio[0]) +")"+ str(now))
