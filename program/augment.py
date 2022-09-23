import numpy as np
import json
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import random
import gc
from preprocess import Preprocess
from labels import Label_gen

from sklearn import model_selection

from tqdm import tqdm

import tensorflow as tf
from keras.backend import tensorflow_backend

user_list = ["user_dr_faisal", "user_dr_habib", "user_dr_habiba", "user_dr_nushrat",
"user_dr_rahman", "user_dr_sajeda", "user_dr_salimul", "user_gn_raisa", "user_hu_nasrin",
"user_hu_rakim", "user_it_shawon", "user_ku_ashiqur", "user_ku_ashir", "user_ku_huda",
"user_ku_kazi", "user_ku_mahmud", "user_ku_maruf", "user_ku_mary", "user_ku_mohasina",
"user_ku_mukta", "user_ku_nuren", "user_ku_nusrat", "user_ku_pias", "user_ku_rafiq",
"user_ku_shakhawat", "user_ku_tania", "user_ku_towfiq"]

class Augment:
    def combine(self, remove):
        data = self.dataset
        labels = self.labels
        # Remove unwanted data and create a list with data and labels together
        if not remove is None:
            dl = [[data[i], labels[i]] for i in range(len(data)) if not labels[i][0] == -1]
        else:
            dl = [[data[i], labels[i]] for i in range(len(data)) if not labels[i] == None]
        del data, labels
        a = [dl[i][0] for i in range(len(dl))]
        b = [dl[i][1] for i in range(len(dl))]
        del dl
        self.dataset = np.array(a)
        self.labels = np.array(b)
        del a, b

    def __init__(self, dataset, labels, num, lan, ratio, remove):
        self.dataset = dataset
        self.labels = labels
        self.Rotate_num = num[0]
        self.Parallel_num = num[1]
        self.Ratio_num = num[2]
        self.ratio = ratio
        self.language = lan
        self.testX = []
        self.testY = []
        self.maxlen = 0
        # label -> None avoid etc.
        self.combine(remove)

    # Separate test and train
    def split(self, u_list, test_num=3, reduce_samples_flg=None, reduce_samples_num=5):
        trainX = []
        trainY = []
        testX = []
        testY = []
        f = open("./data/all/" + self.language + "_uid.txt", "rb")
        uid = pickle.load(f)
        f.close()
        test_user = random.sample(user_list, test_num)
        for i in range(len(self.dataset)):
            if not uid[i] in test_user:
                trainX.append(self.dataset[i])
                trainY.append(self.labels[i])
            else:
                testX.append(self.dataset[i])
                testY.append(self.labels[i])
        self.dataset = trainX
        self.labels = trainY
        trainX = []
        trainY = []
        # When you want to reduce the number of trains
        if not reduce_samples_flg is None:
            for u in test_user:
                u_list.remove(u)
            keep_user = random.sample(u_list, reduce_samples_num)
            for i in range(len(self.dataset)):
                if not uid[i] in keep_user:
                    trainX.append(self.dataset[i])
                    trainY.append(self.labels[i])
        self.testX = testX
        self.testY = testY
        del trainX, trainY, testX, testY

# Functions for SMOTE
    def padding(self, d):
        xdata = sequence.pad_sequences(d, maxlen=self.maxlen, padding = 'post', dtype=np.float32)
        print("done padding")
        del d
        return xdata

    def smote(self):
        sm = SMOTE(random_state=42)
        # l = []
        a = np.empty((len(self.dataset), self.maxlen*6))
        for i, d in enumerate(self.dataset):
            a[i] = np.reshape(d, (self.maxlen*6,))
            # l.append(np.reshape(d, (self.maxlen*6,)))
        self.dataset = a
        self.dataset, self.labels = sm.fit_sample(self.dataset, self.labels)
        b = np.empty(len(self.dataset), (self.maxlen, 6))
        for i, d in enumerate(self.dataset):
            b[i] = np.reshape(d, (self.maxlen, 6))
        self.dataset = b
        # return X_res, Y_res

#So far

    # Translate Stroke

    def genParallel(self, data, label, num):
        print("data augmentation(parallel)")
        a = np.array(data)
        d = []
        labels = []
        for j in tqdm(range(num)):

            ###### reset stroke value to 0 for each word, rewrite random noise value, initialize l

            for i, w in enumerate(a):
                st = 0
                xn = random.gauss(0, 0.001)
                yn = random.gauss(0, 0.001)
                # xn = random.uniform(-per, per)
                # yn = random.uniform(-per, per)
                l = []
                # Put the same noise on one stroke (parallel movement)

                for p in w:
                    # same stroke
                    if p[2] == st:
                        pass
                    # Rewrite noise values ​​when stroke changes
                    else:
                        xn = random.gauss(0, 0.001)
                        yn = random.gauss(0, 0.001)
                        # xn = random.uniform(-per, per)
                        # yn = random.uniform(-per, per)
                        st += 1

                    p0 = p[0] + xn
                    p1 = p[1] + yn

                    # x coordinate, y coordinate, stroke
                    l.append([p0, p1, p[2]])

                labels.append(label[i])
                d.append(l)

        self.dataset = d
        self.labels = labels
        del a, d, labels, l, xn, yn

    #####  genRotate  --> strokeRotate  --> pointRotate
    # point rotation

    def pointRotate(self, point, center, angle):
        # x, y = point
        # c_x, c_y = center
        rad = math.radians(angle)
        # matrix for point rotation

        mat = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        xy = np.array([point[0]-center[0], point[1]-center[1]])
        # point rotation formula

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


    def genRotate(self, data, label, dispersion, num):
        print("data augmentation(rotate)")
        a = np.array(data)
        labels = []
        d = []
        l = []
        cal = 0
        for k in tqdm(range(num)):
            # Set stroke counter to 1 per word

            # Empty c for each word

            # a: [word,,,word]
            for j, w in enumerate(a):
                st = 1
                c = []
                # w: [[x,y,st],,,[x,y,st]]
                # i: index, p: [x,y,st]
                for i, p in enumerate(w):
                    # If the last point in the word, add and rotate

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
                    # If it is not the end point, add it as it is

                    elif w[i+1][2] != st:
                        l.append(p)
                labels.append(label[j])
                d.append(c)
        self.labels = labels
        self.dataset = d
        del a, labels, d, l, c, st


    # wind ratio conversion program
    def genRatio(self, data, label, num, ratio):
        print("data augmentation(XYratio)")
        a = np.array(data)
        d = []
        labels = []
        # Repeat by the magnification of expansion
        for j in tqdm(range(num)):
            ##### Returns a random number for each word #####
            # i : index,  w : word (0(x),1(y),2(stroke))
            for i, w in enumerate(a):
                l = []
                # Mean = [x, y, stroke]
                Mean = np.mean(w, axis=0)
                # 0 <= ran <= 2
                ran = random.randint(0,2)

                for p in w:
                    # Expansion in X direction(1)
                    if ran == 0:
                        if p[0] >= Mean[0]:
                            p0 = p[0] + p[0]*ratio[j]
                        else: #p[0] < Mean[0]
                            p0 = p[0] - p[0]*ratio[j]

                        p1 = p[1]
                    # Expansion in Y direction(2)
                    else:
                        if p[1] >= Mean[1]:
                            p1 = p[1] + p[1]*ratio[j]
                        else: #p[1] < Mean[1]
                            p1 = p[1] - p[1]*ratio[j]

                        p0 = p[0]

                    #x coordinate, y coordinate, stroke
                    l.append([p0, p1, p[2]])

                labels.append(label[i])
                d.append(l)

        self.dataset = d
        self.labels = labels
        del a, l, d, labels, ratio, ran, p0, p1, Mean

    # Not available for image conversion
    def pointToLine(self, data):
        print("converting points to lines")
        l = []
        for c in tqdm(data):
            a = []
            for i in range(len(c) - 1):
                x = c[i][0] # the x-coordinate of the starting point of the line
                y = c[i][1] # the y-coordinate of the starting point of the line
                dx = c[i+1][0] - c[i][0] # 2x-axis distance between points
                dy = c[i+1][1] - c[i][1] # 2y-axis distance between points
                sst = int(c[i][2] == c[i+1][2]) # the same stroke
                dst = int(c[i][2] != c[i+1][2]) # different strokes
                line = np.array([x, y, dx, dy, sst, dst])
                a.append(line)
            a = np.array(a)
            l.append(a)
        self.dataset = np.array(l)
        print("done converting points to lines")

    # Convert the expanded coordinate data to an image
    def Chara_img(self, data, label, type):
        # data = [[word],,,[word]]
        # word = [[x, y, s],,,[x, y, s]]
        # label = [[word_label],,,[word_label]]
        col_num = 2
        row_num = 2
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

        for i in range(4):
            ran = random.randint(0, len(data))
            word = data[ran] #Determining output words
            a = np.array(word)
            st = 0
            d = []
            l = []
            # Separate lines for each stroke
            for index, p in enumerate(a):
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
                if index == len(a)-1:
                    l = np.array(l).T
                    d.append(l)
                    l = []
                    l.append(p)
                    st+=1

            plt.subplot(row_num, col_num, i+1)
            plt.title(str(label[ran]))
            for l in d:
                plt.plot(l[0], l[1], marker = "o", color="r")

            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])

        plt.savefig("./data/all/Kaze/image/"+str(type)+".png")
        plt.close()

# --------------------------------------------------------------------------------

    def runThis(self, augment_mode=0, check_character=None, now=None):
        if check_character is None:
            # Testdata and Train split the data
            self.split(user_list)
            with open("./data/all/test_data.txt", "wb") as fp:   #Pickling
                pickle.dump(self.testX, fp)
            with open("./data/all/test_labels.txt", "wb") as fp:   #Pickling
                pickle.dump(self.testY, fp)
            if augment_mode == 1:
                if self.Rotate_num > 0:
                    self.genRotate(self.dataset, self.labels, dispersion=1.5, num=self.Rotate_num)
                    gc.collect()
                if self.Parallel_num > 0:
                    self.genParallel(self.dataset, self.labels, num=self.Parallel_num)
                    gc.collect()
                if self.Ratio_num > 0:
                    self.genRatio(self.dataset, self.labels, num=self.Ratio_num, ratio=self.ratio)
                    gc.collect()
        # image acquisition
        else:
            self.Chara_img(self.dataset, self.labels, type="NoAug:" + str(now))
            if self.Rotate_num > 0:
                self.genRotate(self.dataset, self.labels, dispersion=15, num=self.Rotate_num)
                gc.collect()
                self.Chara_img(self.dataset, self.labels, type="Rotate:" + str(now))
            if self.Parallel_num > 0:
                self.genParallel(self.dataset, self.labels, num=self.Parallel_num)
                gc.collect()
                self.Chara_img(self.dataset, self.labels, type="Parallel:" + str(now))
            if self.Ratio_num > 0:
                self.genRatio(self.dataset, self.labels, num=self.Ratio_num, ratio=self.ratio)
                gc.collect()
                self.Chara_img(self.dataset, self.labels, type="Ratio:(" + str(self.ratio[1]) +")"+ str(now))


# ----------------------------------------------------------------
if __name__ == '__main__':
    print("Do-Nathing")
