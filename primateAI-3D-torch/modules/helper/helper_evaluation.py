import random
import numpy as np
import torch
import pandas as pd
from helper.helper_data import mapToProtSplitIdxs, mapFromProtSplitIdxs, aa_idx_tup
from helper.helper_file import mkdir_p
import os



def savePreds(newDict, epoch, c):
    basePath = os.path.join(c["runFolder"], "eval", "wholeProteome")
    savePredFile = "%s.epoch%03d.csv.gz" % (basePath, int(epoch))
    print("\t", savePredFile)

    mkdir_p(os.path.dirname(savePredFile))

    dfs = []
    for i, (gene_name, geneDict) in enumerate(newDict.items()):

        scoreArr = geneDict["scores_full"]
        scores_flat = scoreArr.flatten()

        aaNr_flat = np.tile(np.arange(20), (scoreArr.shape[0], 1)).flatten()
        poss_flat = np.repeat(np.arange(0, scoreArr.shape[0]), 20).flatten()

        df_local = pd.DataFrame({"score": scores_flat, "label_numeric_aa_alt": aaNr_flat, "change_position_1based_split": poss_flat, "gene_name_ext": gene_name})
        dfs.append(df_local)

    df = pd.concat(dfs)
    del dfs

    idxToAa = [aa for aa, _ in aa_idx_tup]
    idxToAaDF = pd.DataFrame(list(enumerate(idxToAa)), columns=["label_numeric_aa_alt", "alt_aa"])
    aaPossDF = pd.read_csv(c["eval_aaPossFilePath"], index_col=False)

    df = df.merge(idxToAaDF, on="label_numeric_aa_alt").merge(aaPossDF, on=["gene_name_ext", "change_position_1based_split", "alt_aa"])
    del aaPossDF

    df.sort_values(["gene_name", "change_position_1based", "ref_aa", "alt_aa"]).to_csv(savePredFile, index=False)


def applyModel_allPos(dataGenerator, df, model, pdbDict, c, epoch):
    print("\tPreparing data")
    dataloader_all = torch.utils.data.DataLoader(dataGenerator, num_workers=0, batch_size=None, batch_sampler=None)
    df = df[['gene_name',
             'change_position_1based',
             'change_position_1based_split',
             'posCovered',
             'gene_name_ext']].copy()

    resTuples = []

    print("\tApplying model")
    scoreNameDict = {}
    for i, multiprot in enumerate(dataloader_all):

        if i % 100 == 0:
            print("\tbatch %d/%d" % (i, dataloader_all.dataset.getNBatches()))

        with torch.no_grad():
            multiprot.toDevice_labels()
            _ = model(multiprot)

        for j, subprot in enumerate(multiprot.protList):

            changePoss = subprot.changePoss.detach().cpu().numpy().squeeze()
            for scoreName, scoreTensor in subprot.outputDict["protein"]["score"].items():

                if i == 0:
                    scoreNameDict[scoreName] = scoreTensor.shape[1]

                prot_batch_scores = scoreTensor.detach().cpu().numpy().squeeze()
                resTuple = (scoreName, subprot.gene_name, changePoss, prot_batch_scores)
                resTuples.append(resTuple)

    print("\tConverting to pdbDict")
    newDict = {}
    for i, (scoreName, geneName, changePoss1based, scoreArr) in enumerate(resTuples):

        if i % 10000 == 0: print("\t", i)

        if geneName in newDict:
            geneDict = newDict[geneName]
        else:
            geneDict = {}
            geneLen = pdbDict[geneName]["change_position_1based"].shape[0]
            for scoreName_local, scoreDim in scoreNameDict.items():
                scoreArr_local = np.full((geneLen, scoreDim), -1000, dtype=np.float32)
                geneDict[scoreName_local] = scoreArr_local

            newDict[geneName] = geneDict

        # print(geneDict[scoreName][changePoss1based,:].shape, scoreArr.shape)
        geneDict[scoreName][changePoss1based, :] = scoreArr

    if c['eval_logWholeProteomeScoresFromEvalNr'] != -1 and int(epoch) >= c['eval_logWholeProteomeScoresFromEvalNr']:
        print("\tSaving scores")
        savePreds(newDict, epoch, c)

    print("\tConverting to evalDF")
    geneDFs = []
    for geneName, geneDF in df.drop_duplicates(subset=["gene_name_ext", "change_position_1based_split"]).groupby("gene_name_ext"):
        for scoreName, scoreArr in newDict[geneName].items():
            scoresEval = scoreArr[geneDF.change_position_1based_split.values, :]

            geneDF[scoreName] = scoresEval.tolist()

        geneDFs.append(geneDF)

    newDF = pd.concat(geneDFs)
    print("\tDone")

    return newDF


def posDictsToDF(posDicts):
    gene_names = []
    poss = []

    for posDict in posDicts:
        gene_names.append( posDict["gene_name"] )
        poss.append( posDict["change_position_1based"] )

    df = pd.DataFrame( {"gene_name": gene_names, "change_position_1based": poss} )

    return df


def reduceToProtsInPdbDict(df, pdbDict):

    missingGenes = []
    for gene_name in df.gene_name.values.tolist():
        if not (gene_name in pdbDict or gene_name+"_0" in pdbDict):
            missingGenes.append(gene_name)

    if len(missingGenes) > 0:
        print("=======================================================================================")
        print("== !!! WARNING: Reducing evaluation data because %s genes not found in pdbDict!" % str(missingGenes))
        print("=======================================================================================")

        df = df[ ~df.gene_name.isin(set(missingGenes)) ].copy()

    return df


class EvalPositionCollection():
    def __init__(self, c, pdbDict, dataGenerator):
        self.c = c
        # self.geneNameToPoss = collections.defaultdict(set)
        self.dfs = []
        self.df = None
        self.posDF_sorted_ensemble = None
        self.dataGenerator = dataGenerator
        self.pdbDict = pdbDict
        self.ensemble_suffix = '_ensemble'

    def finishCollection(self):
        #All evaluation positions have been collected; now distribute them into batches that can be used as input for the model
        self.df = pd.concat(self.dfs)
        self.df = mapToProtSplitIdxs(self.df, self.c["input_splitProtIdMappingFilePath"])



    def addPositions_dicts(self, posDict_list):
        df = posDictsToDF(posDict_list)
        self.dfs.append(df)


    def addPositions_df(self, posDF):
        self.dfs.append( posDF )


    def calculateScores(self, model, evalNr):
        #posDictsSorted = self.getPosDicts()
        if not self.df is None:
            with torch.set_grad_enabled(False):
                self.posDF_sorted = applyModel_allPos(self.dataGenerator, self.df, model, self.pdbDict, self.c, evalNr)

                print("\t", "Mapping back positions")
                self.posDF_sorted = mapFromProtSplitIdxs(self.posDF_sorted, self.c)
                print("\t", "Done")
