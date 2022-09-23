import numpy as np
import json
## import requests
import math
import matplotlib
import matplotlib.pyplot as plt
import pickle

DIST = 10
COS = 0.99

class Preprocess:
    def __init__(self):
        self.inks = []
        self.words = []
        self.urls = []
        self.dataset = []

    # Store data in variables in classes
    def importJson(self):
        print("importing data...")
        a = open('./data/export.json')
        b = json.load(a)
        samples = b["medical-samples"]


        self.inks.append(samples[0]["data"]["request"]["inks"])
        self.words.append(samples[0]["word"])
        self.urls.append(samples[0]["imageUrl"])

        print("done importing data")

    #
    # Binary the list and save it as text data
    def saveLists(self, d, name):
        with open("./data/" + name + ".txt", "wb") as fp:   #Pickling
            pickle.dump(d, fp)
    #
    # # Save each variable to a txt file
    # def preprocess(self, lan):
    #     # used for labeling
    #     self.saveLists(self.inks, lan + "_inks")
    #     self.saveLists(self.words,lan + "_words")
    #     self.saveLists(self.urls, lan + "_urls")
    #     self.saveLists(self.uid, lan + "_uid")

    # Function to add stroke number
    def stroke(self):
        for y in range(len(self.inks)):
            for x in range(len(self.inks[y])):
                self.inks[y][x].append([x] * len(self.inks[y][x][0]))

    #Connect Strokes and Transpose
    def strokeConnect(self):
        print("connecting strokes")
        for y in range(len(self.inks)): # Work character by character
            f = self.inks[y][0]
            f = np.array(f).T
            for x in range(len(self.inks[y]) - 1): # Work per stroke
                e = self.inks[y][x+1]
                e = np.array(e).T
                f = np.append(f, e, axis = 0)
            self.dataset.append(f)
        print("done connecting strokes")

    # Remove time information
    def removeTime(self):
        b = []
        for x in range(len(self.dataset)):
            a = np.array(self.dataset[x]).T
            a = np.delete(a, 2, 0)
            a = a.T
            b.append(a)
        self.dataset = b

    # Remove Nearby Points
    def listClosePoint(self):
        print("removing extra points")
        l = []
        for c in self.dataset:
            a = []
            for i in range(len(c)):
                if i == 0 or i+1 == len(c):
                    # leave the first and last points of the character intact
                    a.append(c[i])
                elif c[i-1][2] != c[i][2] or c[i][2] != c[i+1][2]:
                    # Leave the first and last points of the stroke intact
                    a.append(c[i])
                else:
                    dp = math.sqrt((c[i][0]-c[i-1][0])**2 + (c[i][1]-c[i-1][1])**2)
                    if dp >= DIST:
                        a.append(c[i])
            if not a is None:
                l.append(a)
        self.dataset = l

    # Removing Points on Lines
    def listStraightPoint(self):
        l = []
        for c in self.dataset:
            a = []
            for i in range(len(c)):
                if i == 0 or i+1 == len(c):
                    # leave the first and last points of the character intact
                    a.append(c[i])
                elif c[i-1][2] != c[i][2] or c[i][2] != c[i+1][2]:
                    # Leave the first and last points of the stroke intact
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

    # Normalization min-max
    def normalization(self):
        print("normalization")
        a = self.dataset[0]
        for i in range(len(self.dataset) -1):
            a = np.vstack((a, self.dataset[i+1]))

        xma = max(np.array(a).T[0])
        xmi = min(np.array(a).T[0])
        yma = max(np.array(a).T[1])
        ymi = min(np.array(a).T[1])
        ndata = []
        for c in self.dataset:
            n1 = (np.array(c).T[0] - xmi) / (xma - xmi)
            n2 = (np.array(c).T[1] - ymi) / (yma - ymi)
            nor_c = [n1, n2, np.array(c).T[2]]
            nor_c = np.array(nor_c).T
            ndata.append(nor_c)
        self.dataset = ndata
        self.saveLists(self.dataset,"pped_data")
        print("done normalization")

    # convert points to lines
    def pointToLine(self):
        print("converting points to lines")
        l = []
        for c in self.dataset:
            a = []
            for i in range(len(c) - 1):
                    x = c[i][0] # the x-coordinate of the starting point of the line
                    y = c[i][1] # the y-coordinate of the starting point of the line
                    dx = c[i+1][0] - c[i][0] # x-axis distance between two points
                    dy = c[i+1][1] - c[i][1] # y-axis distance between two points
                    sst = int(c[i][2] == c[i+1][2]) # the same stroke
                    dst = int(c[i][2] != c[i+1][2]) # different strokes
                    line = [x, y, dx, dy, sst, dst]
                    a.append(line)
            l.append(a)
        self.dataset = l
        print("done converting points to lines")


    def runThis(self):
        self.importJson()
        # self.preprocess()
        self.stroke()
        self.strokeConnect()
        self.removeTime()
        self.listStraightPoint()
        self.listClosePoint()

        # self.saveLists(self.dataset,"pped_data")
        self.normalization()
        # self.pointToLine()

        print("done preprocessing")
