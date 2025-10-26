import torch
from loss.helper_loss import LossCollection
from workflow_training import ScoreCollection

class EvaluationValidation():
    def __init__(self, dataloader, lossObj, c, maxProts = -1):
        self.dataloader = dataloader
        self.lossObj = lossObj
        self.maxProts = maxProts
        self.c = c
        self.scoreCollection = ScoreCollection(None)

    def evaluate(self, model):
        lossCollection = LossCollection()
        with torch.set_grad_enabled(False):

            for i, multiprot in enumerate(self.dataloader):

                if i % 100 == 0: print(i)

                multiprot.toDevice_labels()
                _ = model(multiprot)

                self.scoreCollection.addMultiprotScores(multiprot)

                lossDict = self.lossObj(self.scoreCollection)

                self.scoreCollection.empty()

                lossCollection.addDict(lossDict)



                #losses.append(lossVal.item())

                if i == self.maxProts and self.maxProts > 0:
                    # print("Val batch loss", i, lossVal.item())
                    # print("Val batch acc", i, acc)
                    break

        #meanLoss = sum(losses) / len(losses)

        #

        print("Validation loss: %s" % (lossCollection.toString()) )

        metricDict = {}


        for metric, val in lossCollection.toTupleList():
            metric = "val_" + metric

            # if self.tboard != None:
            #     self.tboard.add_scalar(metric, val, epoch)
            metricDict[metric] = val


        return metricDict


