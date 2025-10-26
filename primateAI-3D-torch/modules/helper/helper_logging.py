import os
import errno
import pandas as pd
from helper.helper_file import mkdir_p
import json

__author__ = 'thamp'

class NnLogger:
    def __init__(self, c, tboard):
        self.c = c
        self.tboard = tboard
        self.logFilePath = os.path.join(c["runFolder"], c["configName"] + ".log.csv")
        self.currLineDict = {}
        self.lineDicts = []
        self.epoch = 0

        self.debugDfFilePathPrefix = os.path.join(self.c["runFolder"], "debug", "debug")
        self.debugDFs = []
        self.debugDFsAdded = 0

    def addList(self, listi):
        for keyi, vali in listi:
            self.currLineDict[keyi] = vali

    def addVal(self, keyi, vali):
        self.currLineDict[keyi] = vali

    def addDict(self, dicti):
        for keyi, vali in dicti.items():
            self.currLineDict[ keyi ] = vali

    def addDebugDFs(self, debugDFs):
        #print("current debug DFs: %d; new: %d" % ( self.debugDFsAdded, len(debugDFs[ self.debugDFsAdded: ]) ))
        self.debugDFs.extend(debugDFs[ self.debugDFsAdded: ])
        self.debugDFsAdded += len(debugDFs[ self.debugDFsAdded: ])

    def getCurrDF(self):
        self.currLineDict["epoch"] = self.epoch
        return self.currLineDict

    def writeDebugDF(self, epoch):
        print("Writing debug log data...")
        outputFilePath = "%s.epoch%03d.pkl" % ( self.debugDfFilePathPrefix, epoch )
        debugDF = pd.concat(self.debugDFs)
        mkdir_p(os.path.dirname(outputFilePath))
        debugDF.to_pickle(outputFilePath)
        print("Done")

    def clearDebugDFs(self):
        self.debugDFs = []

    def finishEpoch(self):
        if self.c["loss_debug_writeScoresDF"] == "perEval":
            self.writeDebugDF(self.epoch)
        self.clearDebugDFs()

        currDF = self.getCurrDF()

        self.lineDicts.append(currDF)

        outputDF = pd.DataFrame(self.lineDicts)
        mkdir_p(os.path.dirname(self.logFilePath))
        outputDF.to_csv(self.logFilePath)

        if self.tboard != None:

            for keyi, itemi in currDF.items():
                self.tboard.add_scalar(keyi, itemi, self.epoch)

        self.currLineDict = {}
        self.epoch += 1
