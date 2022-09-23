# Data image acquisition
import datetime
import prepro as pp
from aug import Augment
import pickle

import numpy as np
import gc

# get parameter
# pa = [[ro_num, ps_num, ra_num], [lr, dout, b_size], ratio(ex:[0.02*10])]
ro = int(input("Stroke Rotate  (Do:1):"))
ps = int(input("Parallel Shift (Do:1):"))
ra = int(input("Change Ratio(X:1 Y:2):"))
ratio = int(input("Ratio　[0.02]→(1~9) :"))
pa = [[ro, ps, ra], [0], [0.02*ratio]]
pp = pp.Preprocess()
pp.runThis()

# data acquisition
f = open("./data/pped_data.txt", "rb")
dataset = pickle.load(f)
f.close()

# get execution time
dt_n = datetime.datetime.now()
now = dt_n.strftime('%m-%d_%H%M')
# Expansion
ag = Augment(dataset=dataset, num=pa[0], ratio=pa[2])
ag.runThis(now=now)
