# データ画像取得
import datetime
import prepro as pp
from aug import Augment
import pickle

import numpy as np
import gc

# パラメータ取得
# pa = [[ro_num, ps_num, ra_num], [lr, dout, b_size], ratio(ex:[0.02*10])]
ro = int(input("Stroke Rotate  (Do:1):"))
ps = int(input("Parallel Shift (Do:1):"))
ra = int(input("Change Ratio(X:1 Y:2):"))
ratio = int(input("Ratio比率　[0.02]→(1~9) :"))
pa = [[ro, ps, ra], [0], [0.02*ratio]]
pp = pp.Preprocess()
pp.runThis()

# データ取得
f = open("./data/pped_data.txt", "rb")
dataset = pickle.load(f)
f.close()

# 実行時間取得
dt_n = datetime.datetime.now()
now = dt_n.strftime('%m-%d_%H%M')
# 拡張
ag = Augment(dataset=dataset, num=pa[0], ratio=pa[2])
ag.runThis(now=now)
