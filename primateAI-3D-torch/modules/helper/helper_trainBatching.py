import random

import numpy as np
import torch
import math
import collections

from helper.helpers import chunks
from helper.helper_subProt import SubProt
from helper.helper_multiprot import MultiProt


class ProtPosDataPreproc():
    def __init__(self, c, pdbDict, geneNameToId, idToGeneName):

        self.c = c
        self.pdbDict = pdbDict
        self.geneNameToId = geneNameToId
        self.idToGeneName = idToGeneName

        print("Collecting positions")
        self.protDictDict = self.getAllProtPoss()

        # valBatches, self.valPosDict = self.getValidationPoss()
        # self.batchesRaw["val"] = valBatches

        self.batchesRaw = {}
        print("Batching all positions")
        self.batchesRaw["all"], self.batchesRaw["allAugm"] = self.getPossBatches()

        print("Batching validation data")
        self.valBatches, self.valPosDict = self.getValidationPoss(self.batchesRaw["all"])
        self.batchesRaw["val"] = self.valBatches

        print("Validation batches: %d out of %d" % ( len(self.batchesRaw["val"]), len(self.batchesRaw["all"]) ))

        #testProtein = list(self.valPosDict.items())[0][0]

        print("Batching training data")
        self.batchesRaw["train"], _ = self.getPossBatches(valPosDict=self.valPosDict)

    def getValidationPoss(self, possBatches):

        valBatches = []

        valPosDict = collections.defaultdict(set)

        nBatches = len(possBatches)

        nBatchesVal = round( nBatches * self.c["eval_validationFraction"] )

        for gene_name_ext, batchLen, poss in possBatches[:nBatchesVal]:
            valPosDict[gene_name_ext.split("_")[0]] = set(poss)
            valBatches.append((gene_name_ext, len(poss), poss))

        return valBatches, valPosDict

    def getAllProtPoss(self):
        protDictDict = {}

        for i, (geneName, p) in enumerate(self.pdbDict.items()):
            #print(geneName)
            #p = self.pdbDict[geneName]

            if i % 1000 == 0: print(i)

            allChangePos1Based = p["change_position_1based"][1:].squeeze().long()

            protDict = {}
            protDict["gene_name"] = geneName

            vec_pos1based = np.concatenate([np.array([-1]), allChangePos1Based])
            # protDict["vec_refAa"] = np.concatenate([np.array([-1]), allRefAAs])
            # protDict["vec_refAaNum"] = np.concatenate([np.array([-1]), allRefAAsNum])

            #assert (vec_pos1based.shape[0] - 1 == vec_pos1based[-1]), print(allChangePos1Based, vec_pos1based.shape, vec_pos1based[-10::])
            assert vec_pos1based.shape[0] == p["feat_coveredRefDF"].shape[0]

            protDict["vec_pos1based"] = vec_pos1based

            protDict["feat_coveredRefDF"] = p["feat_coveredRefDF"]

            protDictDict[ geneName ] = protDict

        return protDictDict

    def getCoveredPoss(self, gene_name):
        protDict = self.protDictDict[gene_name]

        vec_pos1based = protDict["vec_pos1based"]
        coveredRefDF = protDict["feat_coveredRefDF"][:, 0] == 1.0

        assert np.max(vec_pos1based[coveredRefDF]) < 100000

        possSet = set(vec_pos1based[coveredRefDF].tolist())

        return possSet

    def getPossBatches(self, valPosDict = None):
        trainBatches_smallExcl = []
        trainBatches_augm = []

        minBatchSize = int(self.c["batch_posPerProt"]) / 2

        prots = list(self.protDictDict.keys())
        random.shuffle(prots)

        skipped = []

        for gene_name in prots:
            possSet = self.getCoveredPoss(gene_name)

            if valPosDict != None:

                if gene_name in valPosDict:

                    #before = len(possSet)

                    possSet = possSet - valPosDict[gene_name]

                    #after = len(possSet)
                    #diff = before - after
                    #if diff > 0: print(before, after, diff)

            if len(possSet) == 0:
                skipped.append( gene_name )
                continue

            possArr = np.array(list(possSet))
            np.random.shuffle(possArr)
            poss = possArr.tolist()

            batches = chunks(poss, self.c["batch_posPerProt"])

            for i, batch in enumerate(batches):

                batch_augm = batch
                if len(batch) < minBatchSize:
                    #print( minBatchSize, len(batch), int(minBatchSize - len(batch)) )
                    extraPoss = random.choices( poss, k=int(minBatchSize - len(batch)) )
                    batch_augm = batch + extraPoss

                if len(batch) >= minBatchSize:
                    trainBatches_smallExcl.append((gene_name + ("#%d" % i), len(batch), batch))

                if len(batch_augm) >= minBatchSize:
                    trainBatches_augm.append((gene_name + ("#%d" % i), len(batch_augm), batch_augm))

                else:
                    pass#print("dropped")

        random.shuffle(trainBatches_smallExcl)
        random.shuffle(trainBatches_augm)

        print("Skipped %s" % len(skipped))

        return trainBatches_smallExcl, trainBatches_augm

    def resetForNewEpoch(self):
        self.batchesRaw["train"], _ = self.getPossBatches(valPosDict=self.valPosDict)


class ProtPossBatchGenerator(torch.utils.data.IterableDataset):

    def __init__(self, datasetName, c, protPosDataPreproc, pdbDict):
        super(ProtPossBatchGenerator).__init__()
        self.datasetName = datasetName
        self.pdbDict = pdbDict
        self.geneNameToId = protPosDataPreproc.geneNameToId



        self.subProtBatchStartIdx = 0
        self.subProtBatchEndIdx = len(protPosDataPreproc.batchesRaw[datasetName])-1

        self.c = c
        self.protPosDataPreproc = protPosDataPreproc

        self.DONOTUSE_allMultiprotBatches = self.getAllBatches()


    def getNBatches(self):
        return len(self.DONOTUSE_allMultiprotBatches)

    def getMultiprotFromBatch(self, batch):

        currProts = []
        for gene_name, poss in batch:

            possArr = np.array(poss) #torch.tensor(poss, device=self.c["torch_device"])

            prot = SubProt(self.c,
                           self.pdbDict[gene_name],
                           gene_name,
                           possArr)

            currProts.append(prot)

        multiprot = MultiProt(self.c, currProts)

        return multiprot


    def getAllBatches(self):

        batchesRaw = self.protPosDataPreproc.batchesRaw[self.datasetName]

        currProts = []

        allMultiprotBatches = []

        for i in range(self.subProtBatchStartIdx, self.subProtBatchEndIdx+1):

            gene_name_ext, batchSize, poss = batchesRaw[i]
            gene_name = gene_name_ext.split("#")[0]

            currProts.append( (gene_name, poss) )

            #TODO destroy singleprots once multiprot is constructed
            if (len(currProts) == self.c["batch_protsPerBatch"]) or (i == self.subProtBatchEndIdx):
                allMultiprotBatches.append(currProts)

                currProts = []

        return allMultiprotBatches


    def __iter__(self):

        allMultiprotBatches = self.getAllBatches()

        #batchesRaw = self.protPosDataPreproc.batchesRaw[self.datasetName]

        for multiprotBatch in allMultiprotBatches:
            yield self.getMultiprotFromBatch(multiprotBatch)



    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()

        dataset = worker_info.dataset  # the dataset copy in this worker process

        overall_start = dataset.batchStartIdx
        overall_end = dataset.batchEndIdx

        per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))

        worker_id = worker_info.id

        dataset.batchStartIdx = overall_start + worker_id * per_worker

        dataset.batchEndIdx = min(dataset.start + per_worker, overall_end)


    def resetForNewEpoch(self):
        self.protPosDataPreproc.resetForNewEpoch()

