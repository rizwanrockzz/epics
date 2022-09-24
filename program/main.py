import datetime
import imp
import preprocess as pp
import labels as la
from learning import Learning
from augment import Augment
import pickle
import sys
import numpy as np
import tensorflow as tf
from keras.backend import backend
# from tf.compat.v1.keras.backend import set_session
import gc


def cleanGpu():
    # GPU Needed to avoid using up memory
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    backend.set_session(session)

# -----------------------------------------------------
# read write For
# before to write what you did last time (at the function level)
# Other than that, include the result and parameters.


def read_log(i):
    print("【Previous execution record】")
    if i == 0:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\log.txt", "r") as f:
            s = f.read()
            print(s)
    elif i == 1:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\main_acc.txt", "r") as f:
            s = f.read()
            print(s)
    elif i == 2:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\parameter.txt", "r") as f:
            s = f.read()
            print(s)
    elif i == 3:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\img.txt", "r") as f:
            s = f.read()
            print(s)
    elif i == 4:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\pp_label.txt", "r") as f:
            s = f.read()
            print(s)


def write_log(i, text):
    # Get Execution Time
    dt_now = datetime.datetime.now()
    dt = dt_now.strftime('%Y-%m-%d_%H: ')
    text = dt + text

    if i == 0:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\log.txt", "a") as f:
            f.write(text + "\n")
    elif i == 1:
        print("Learning Already filled in")
    elif i == 2:
        print("Learning Already filled in")
    elif i == 3:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\img.txt", "a") as f:
            f.write(text + "\n")
    elif i == 4:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\pp_label.txt", "a") as f:
            f.write(text + "\n")


def delete(i, text):
    # Get Execution Time
    dt_now = datetime.datetime.now()
    dt = dt_now.strftime('%Y-%m-%d_%H: ')
    text = dt + text

    # f  : Original record (erase this and make it new)
    # f2 : A file that keeps the original record
    # f3 : Write a new to f

    if i == 5:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\log.txt", "r") as f:
            s = f.read()
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\log-data.txt", "a") as f2:
            f2.write(s)
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\log.txt", "w") as f3:
            f3.write(text)
    elif i == 1:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\main_acc.txt", "r") as f:
            s = f.read()
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\main_acc-data.txt", "a") as f2:
            f2.write(s)
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\main_acc.txt", "w") as f3:
            f3.write(text)
    elif i == 2:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\parameter.txt", "r") as f:
            s = f.read()
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\parameter-data.txt", "a") as f2:
            f2.write(s)
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\parameter.txt", "w") as f3:
            f3.write(text)
    elif i == 3:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\img.txt", "r") as f:
            s = f.read()
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\img-data.txt", "a") as f2:
            f2.write(s)
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\img.txt", "w") as f3:
            f3.write(text)
    elif i == 4:
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\pp_label.txt", "r") as f:
            s = f.read()
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\pp_label-data.txt", "a") as f2:
            f2.write(s)
        with open(r"D:\epics\datasets\program\data\all\Kaze\log\pp_label.txt", "w") as f3:
            f3.write(text)
# ----------------------------------------------------------
# Input functions


def Input(last_num):
    while True:
        choose = int(input("input："))
        if 1 <= choose <= last_num:
            break
    return choose
# -----------------------------------------------------------


def Parameter(learning_flag):
    # Result
    result = []
    # Determining the Expansion Factor
    ro_num = int(input("Stroke-Rotate  num :"))
    ps_num = int(input("Paralell-Shift num :"))
    ra_num = int(input("XY Ratio       num :"))
    num = [ro_num, ps_num, ra_num]
    result.append(num)
    # for parameter validation
    if learning_flag:
        print("learning rate, dropout, batch sizing")
        lr = float(input("lr(0.001) :"))
        dout = float(input("dout(0.3) :"))
        b_size = int(input("batch_size(512) :"))
        learn = [lr, dout, b_size]
    else:
        lr = 0.001
        dout = 0.3
        b_size = 512
        learn = [lr, dout, b_size]
    result.append(learn)
    # Ratio
    print("Determine the baseline for the ratio expansion")
    print("(1)Default(0.02), (2)0.005, (3)0.01, (4)0.03, (5)0.04")
    n = Input(5)
    # Determining the magnification
    ra_ratio = [0, 0.02, 0.005, 0.01, 0.03, 0.04]
    ratio = [a * ra_ratio[n] for a in range(ra_num)]
    if ra_num <= 1:
        ratio = [0, 0]
    result.append(ratio)

    return result
# ------------------------------------------------------------
# Remove Memo


def fun_5():
    print("Choose Remove Text")
    print("1-Machine Learning Execution")
    print("2-Parameter Validation")
    print("3-Data image acquisition")
    print("4-Pre-treated & labeled")
    print("5-Before All Log")
    i = Input(5)
    delete(i, text="Delete")

# Pre-treated & labeled ...


def fun_1():
    print("Machine Learning Execution")
    # Output of results up to the last time
    # read_log(1) #commenting read_log for now
    read_log(1)
    print("1-All words 2-English 3-The Bangla language")
    lan = ["0", "all", "en", "bn"]    # @ TO BE EDITED LANGUAGE *&^%^&
    choose_lan = Input(3)
    # Parameter Acquisition
    # pa = [[ro_num, ps_num, ra_num], [lr, dout, b_size], ratio(ex:[0.02*10])]
    pa = Parameter(learning_flag=True)

    f = open("./data/all/" + lan[choose_lan] + "_pped_data.txt", "rb")
    dataset = pickle.load(f)
    f.close()
    # Word labels for learning
    f = open("./data/all/" + lan[choose_lan] + "_bined_labels.txt", "rb")
    labels = pickle.load(f)
    f.close()
    # expansion
    ag = Augment(dataset=dataset, labels=labels,
                 num=pa[0], lan=lan[choose_lan], ratio=pa[2], remove=-1)
    ag.runThis(augment_mode=1, check_character=None, now=None)
    ag.pointToLine(ag.dataset)
    dataset = ag.dataset
    labels = ag.labels

    f = open("./data/all/test_data.txt", "rb")
    testX = pickle.load(f)
    f.close()
    f = open("./data/all/test_labels.txt", "rb")
    testY = pickle.load(f)
    f.close()
    ag.pointToLine(testX)
    testX = ag.dataset
    del ag
    gc.collect()

    ml = Learning(dataset, labels, testX, testY)
    del dataset, labels, testX, testY
    gc.collect()

    cleanGpu()
    ml.runThis()  # Shape your data
    cleanGpu()
    # pa = [[ro_num, ps_num, ra_num], [lr, dout, b_size], ratio(ex:[0.02*10])]
    text = lan[choose_lan] + ":num("+str(pa[0][0])+","+str(pa[0][1])+","+str(
        pa[0][2])+"):parameter:(" + str(pa[1][0])+","+str(pa[1][1])+","+str(pa[1][2])+"):ratio:"+str(pa[2][1])
    name = "main_acc.txt"
    ml.recogModel(hidden_size=300, dout=pa[1][1], rc_dout=0.3,
                  dense_unit=200, optName="adam", lr=pa[1][0])
    ml.train(b_size=pa[1][2], epcs=5, split=0.1)
    loss, accuracy = ml.eval(ml.testX, ml.testY, text, name)
    gc.collect()

    # Get Writes→ Writes
    # write_log(i=2, text=text)

# Parameter Validation


def fun_2():
    print("Parameter Validation")
    # Output of results up to the last time
    # read_log(2) #commenting read_log for now
    read_log(2)
    print("1-All words 2-English 3-Bangla")
    lan = ["0", "all", "en", "bn"]
    choose_lan = Input(3)
    # Parameter Acquisition
    # pa = [[ro_num, ps_num, ra_num], [lr, dout, b_size], ratio(ex:[0.02*10])]
    pa = Parameter(learning_flag=True)

    f = open("./data/all/" + lan[choose_lan] + "_pped_data.txt", "rb")
    dataset = pickle.load(f)
    f.close()
    # Word labels for learning
    f = open("./data/all/" + lan[choose_lan] + "_bined_labels.txt", "rb")
    labels = pickle.load(f)
    f.close()
    # expansion
    ag = Augment(dataset=dataset, labels=labels,
                 num=pa[0], lan=lan[choose_lan], ratio=pa[2], remove=-1)
    ag.runThis(augment_mode=1, check_character=None, now=None)
    ag.pointToLine(ag.dataset)
    dataset = ag.dataset
    labels = ag.labels

    f = open("./data/all/test_data.txt", "rb")
    testX = pickle.load(f)
    f.close()
    f = open("./data/all/test_labels.txt", "rb")
    testY = pickle.load(f)
    f.close()
    ag.pointToLine(testX)
    testX = ag.dataset
    del ag
    gc.collect()

    ml = Learning(dataset, labels, testX, testY)
    del dataset, labels, testX, testY
    gc.collect()

    cleanGpu()
    ml.runThis()  # Shape your data
    cleanGpu()
    # pa = [[ro_num, ps_num, ra_num], [lr, dout, b_size], ratio(ex:[0.02*10])]
    text = lan[choose_lan] + ":num("+str(pa[0][0])+","+str(pa[0][1])+","+str(
        pa[0][2])+"):parameter:(" + str(pa[1][0])+","+str(pa[1][1])+","+str(pa[1][2])+"):ratio:"+str(pa[2][1])
    name = "parameter.txt"
    ml.recogModel(hidden_size=300, dout=pa[1][1], rc_dout=0.3,
                  dense_unit=200, optName="adam", lr=pa[1][0])
    ml.train(b_size=pa[1][2], epcs=5, split=0.1)
    loss, accuracy = ml.eval(ml.testX, ml.testY, text, name)
    gc.collect()

    # Get Writes→ Writes
    # write_log(i=2, text=text)

# Data image acquisition


def fun_3():
    print("Data image acquisition")
    # Output of results up to the last time
    # read_log(3) #commenting read_log for now
    read_log(3)
    print("1-All words 2-English 3-Bangla")
    lan = ["0", "all", "en", "bn"]
    choose_lan = Input(3)
    # Parameter Acquisition
    # pa = [[ro_num, ps_num, ra_num], [lr, dout, b_size], ratio(ex:[0.02*10])]
    pa = Parameter(learning_flag=False)
    # Data Acquisition
    f = open("./data/all/" + lan[choose_lan] + "_pped_data.txt", "rb")
    dataset = pickle.load(f)
    f.close()
    # Word labels for titles
    f = open("./data/all/" + lan[choose_lan] + "_word_labels.txt", "rb")
    labels = pickle.load(f)
    f.close()
    # Get Execution Time
    dt_n = datetime.datetime.now()
    now = dt_n.strftime('%m-%d_%H')
    # expansion
    ag = Augment(dataset=dataset, labels=labels,
                 num=pa[0], lan=lan[choose_lan], ratio=pa[2], remove=None)
    ag.runThis(augment_mode=1, check_character=1, now=now)

    # Get Writes→ Writes
    text = lan[choose_lan] + ":num("+str(pa[0][0])+"," + \
        str(pa[0][1])+","+str(pa[0][2])+"):ratio:"+str(pa[2][1])
    write_log(i=3, text=text)

# Pre-treated & labeled


def fun_4():
    print("Pre-treated & labeled")
    # Output of results up to the last time
    # read_log(4) #commenting read_log for now
    read_log(4)

    print("1-All words 2-English 3-Bangla")
    lan = ["0", "all", "en", "bn"]
    choose_lan = Input(3)

    print("1- Learning label 2-Label for image title")
    label = ["0", "learning", "image-title"]
    choose_label = Input(2)

    pp.preprocess_main(lan[choose_lan])
    la.labels_main(lan[choose_lan], label[choose_label])

    # Get Writes→ Writes
    text = lan[choose_lan] + ":" + label[choose_label]
    write_log(i=4, text=text)


# ---------------------------------------------------
if __name__ == '__main__':
    # Output of previous execution result
    # read_log(0) #commenting read_log for now
    read_log(0)
    # Action Selection
    # Machine learning and parameter verification are almost the same, only the output destination is different
    print("Please select an action")
    print("1. Machine Learning Execution")
    print("2. Parameter Validation")  # Validation And
    print("3. Data image acquisition")
    print("4. Pre-treated & labeled")
    print("5. Remove Memory")
    fun = Input(5)
    name = ["0", "Machine Learning Execution", "Parameter Validation",
            "Data Image Acquisition", "Preprocessing & Labeled", "Remove"]
    # Write to last execution record
    write_log(i=0, text=name[fun])

    # Function Execution
    if name[fun] == "Remove":
        fun_5()
    elif name[fun] == "Machine Learning Execution":
        fun_1()
    elif name[fun] == "Parameter Validation":
        fun_2()
    elif name[fun] == "Data image acquisition":
        fun_3()
    elif name[fun] == "Pre-treated & labeled":
        fun_4()
    else:
        print("An error occurred during function execution.")
        sys.exit()

    print("Dooooone!")
