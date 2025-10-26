import time
import pandas as pd
import torch
import torch.multiprocessing as mp
import os
import math

from helper.helper_file import mkdir_p
from loss.helper_loss import LossCollection

class TrainingWorkflow():

    def __init__(self, model, lossObj, optimizer, logger, evals, valEval,
                 evalPositionCollection):
        self.model = model
        self.lossObj = lossObj
        self.optimizer = optimizer
        self.logger = logger
        self.valEval = valEval
        self.evals = evals
        self.c = model.c
        self.batchCounter = 0
        self.evalCounter = 0
        self.nGradients = 0

        self.lossCollection = LossCollection()
        self.scoreCollection = ScoreCollection(self.c)

        self.evalPositionCollection = evalPositionCollection
        self.total_time_start = time.time()



    def saveCheckpoint(self, epochNr):
        checkpointOutputFilePath = os.path.join( self.c["runFolder"], "checkpoints", "%03d.tar" % epochNr )
        mkdir_p(os.path.dirname(checkpointOutputFilePath))

        print(checkpointOutputFilePath)

        torch.save({
            'epochNr': epochNr,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.lossObj.state_dict()
        }, checkpointOutputFilePath)


    def trainEpoch(self, dataloader, epoch, nBatches=None, nEvalsPerEpoch=1, nBatchesPerTrainEval=100, saveCheckpoint=True):

        _ = self.model.train()

        if nBatches == None:
            nBatches = 10000000000

        start_time = time.time()

        nBatchesPerEval = int(math.floor(dataloader.dataset.getNBatches() / float(nEvalsPerEpoch)))
        for i, multiprot in enumerate(dataloader):

            multiprot.toDevice_labels()
            _ = self.model(multiprot)

            self.scoreCollection.addMultiprotScores(multiprot)

            lossDict = self.lossObj(self.scoreCollection)
            self.scoreCollection.empty()
            lossVal = lossDict["loss"]
            self.lossCollection.addDict(lossDict)
            self.logger.addDebugDFs(self.lossObj.dfs)

            lossVal.backward()
            self.nGradients += 1
            self.optimizer.step()
            self.nGradients = 0

            self.optimizer.zero_grad()

            if i % nBatchesPerTrainEval == 0 and i > 0:
                end_time = time.time()

                print("Batch %d of %d; time (last 100): %.1f; batch losses: %s" % (i, dataloader.dataset.getNBatches(), end_time - start_time, self.lossCollection.toString()))

                start_time = time.time()


            if (self.batchCounter+1) % nBatchesPerEval == 0 and self.batchCounter > 0:
                evalStart = time.time()

                print("Total batch %d; Evaluating..." % (self.batchCounter))

                _ = self.model.eval()
                _ = self.lossObj.eval()

                with torch.no_grad():
                    self.evaluateEpoch(self.evalCounter, self.batchCounter, epoch, self.lossCollection)

                _ = self.model.train()
                _ = self.lossObj.train()

                self.evalCounter += 1
                self.lossCollection.reset()

                print("Loss collection")
                print(self.lossCollection.toString())

                print("Evaluation done in %d" % (time.time() - evalStart))

            self.batchCounter += 1

            if i == nBatches:
                break

        if saveCheckpoint:
            print("Saving checkpoint")
            self.saveCheckpoint(epoch)

        dataloader.dataset.resetForNewEpoch()

        if self.c["loss_debug_writeScoresDF"] == "perEpoch":
            self.logger.writeDebugDF(epoch)
            self.logger.clearDebugDFs()

        return None#meanLoss



    def evaluateEpoch(self, evalCounter, batchCounter, epochNr, lossCollection):

        t = time.time()

        self.logger.addList(lossCollection.toTupleList())
        self.logger.addVal("evalNr", evalCounter)
        self.logger.addVal("batchNr", batchCounter)
        self.logger.addVal("epochNr", epochNr)
        self.logger.addVal("time(s)", time.time() - self.total_time_start)

        self.evalPositionCollection.calculateScores(self.model, evalCounter)
        print("Predicting eval positions in %d" % (time.time() - t))

        t = time.time()
        valMetricsDict = self.valEval.evaluate(self.model)
        print("Validation in %d" % (time.time() - t))

        t = time.time()
        metricsDicts = evalHelperHelper(self.evals, evalCounter)
        print("Stats in %d" % (time.time() - t))

        metricsDicts.append(valMetricsDict)

        for metricDict in metricsDicts:
            self.logger.addDict(metricDict)

        pd.set_option('display.max_rows', 500)
        print( pd.Series(self.logger.getCurrDF()).sort_index() )

        self.logger.finishEpoch()


def evalHelper(evalObj_model_epoch, q):
    evalObj, epoch = evalObj_model_epoch
    metricDict = evalObj.evaluate(epoch)
    q.put(metricDict)

    #return metricDict

def evalHelperHelper(evaluations, evalCounter):
    metricDicts = []

    # run non-parallel evals
    for eval in evaluations:
        metrics_eval = eval.evaluate(evalCounter)
        metricDicts.append(metrics_eval)

    return metricDicts

class ScoreCollection():

    def __init__(self, c):
        self.c = c
        self.subprot_outputs = []
        self.nMultiprots = 0

    def addMultiprotScores(self, multiprot):
        for subprot in multiprot.protList:
            self.subprot_outputs.append( subprot.outputDict )

        self.nMultiprots += 1

    def empty(self):
        self.subprot_outputs = []
        self.nMultiprots = 0

