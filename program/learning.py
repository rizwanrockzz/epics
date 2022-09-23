## -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, InputLayer, SimpleRNN, Activation, Flatten, Reshape, Dropout, AveragePooling1D
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import pickle

from keras.engine.topology import Input
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop

from preprocess import Preprocess
from labels import Label_gen
from augment import Augment

import datetime
import os

import tensorflow as tf
from keras.backend import tensorflow_backend

class Learning:
    def __init__(self, data, labels, testX, testY):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.maxlen = 0
        self.testX = np.array(testX)
        self.testY = np.array(testY)
        self.model = 0
        self.dt = 0
        self.history = 0

    def padding(self, d):
        xdata = sequence.pad_sequences(d, maxlen=self.maxlen, padding = 'post', dtype=np.float32)
        print("done padding")
        del d
        return xdata

    def recogModel(self, hidden_size=500, dout=0.2, rc_dout=0.3, dense_unit=200, optName="adam", lr=0.01):
        _input = Input(shape=(self.maxlen, 6))
        x = Bidirectional(LSTM(hidden_size, recurrent_dropout=rc_dout), merge_mode='ave')(_input)
        x = Dropout(dout)(x)
        x = Reshape((hidden_size,1))(x)
        x = AveragePooling1D(pool_size=2)(x)
        x = Reshape((int(hidden_size/2),))(x)
        x = Dropout(dout)(x)
        x = Dense(dense_unit)(x)
        predictions = Dense(480, activation='softmax')(x)
        model = Model(inputs=_input, outputs=predictions)
        if optName.lower() == "adam":
            opt = Adam(lr=lr)
            #lr = 0.001がデフォルト
        elif optName.lower() == "sgd":
            opt = SGD(lr=lr)
            #lr = 0.01がデフォルト
        elif optName.lower() == "rmsprop":
            opt = RMSprop(lr=lr)
            #lr = 0.001がデフォルト
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        print(model.summary())
        dt_now = datetime.datetime.now()
        self.dt = dt_now.strftime('%Y-%m-%d_%H-%M')
        json_string = model.to_json()
        open(os.path.join("./models", self.dt+"_model.json"), 'w').write(json_string)
        self.model = model

    def train(self, b_size=32, epcs=100, split=0.1, es_patience=0):
        self.history = self.model.fit(self.data, self.labels, batch_size=b_size, epochs=epcs, validation_split=split, callbacks=[EarlyStopping(patience=0)])

    def eval(self, testX, testY, text, name):
        loss, accuracy = self.model.evaluate(testX, testY)
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)
        with open("./data/all/Kaze/log/" + name, "a") as fp:
            fp.write("title : " + str(text)+"\n")
            fp.write("Date  : " + str(self.dt)+"\n")
            fp.write('Test loss:' + str(loss) + ' Test accuracy:' + str(accuracy)+"\n")

        print('show 学習曲線')
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        self.plot_history_loss(fig, axL, axR)
        self.plot_history_acc(fig, axL, axR)
        fig.savefig("./data/all/Kaze/acc-fig/"+self.dt+ "(" + text + ").png")
        plt.close()
        return loss, accuracy

    def plot_history_loss(self, fig, axL, axR):
        axL.plot(self.history.history['loss'],label="loss for training")
        axL.plot(self.history.history['val_loss'],label="loss for validation")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='upper right')
    # acc
    def plot_history_acc(self, fig, axL, axR):
        # Plot the loss in the history
        axR.plot(self.history.history['acc'],label="loss for training")
        axR.plot(self.history.history['val_acc'],label="loss for validation")
        axR.set_title('model accuracy')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc='upper right')

    def runThis(self):
        # 学習用のデータの形を整える
        self.maxlen = max([len(self.data[i]) for i in range(len(self.data))])
        self.data = self.padding(self.data)
        # 評価用データの形を整える
        self.testX = self.padding(self.testX)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Do-Nathing")
