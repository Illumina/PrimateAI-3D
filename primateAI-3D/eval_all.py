from nn_worker_helper import  data_generation_triples
from keras.callbacks import Callback

import pandas as pd
import numpy as np
import os

import time
from voxelization_accel_helpers import getNFeats, voxelizeFromTriples
import lmdb

class AllEval(Callback):
    def __init__(self, c, variantsDfFilePath, pdbLmdb_path, verbose=0):
        super(AllEval, self).__init__()

        pdbLmdb = lmdb.open(pdbLmdb_path, create=False, subdir=True, readonly=True, lock=False)

        allDF = pd.read_csv(variantsDfFilePath, index_col=0)

        if not "name" in allDF.columns:
            allDF["name"] = allDF["gene_name"]

        allDF = allDF[~allDF["name"].isna()].copy()

        allRows_tmpDF = allDF[["name", "change_position_1based"]].drop_duplicates()# #

        allRows = []
        allRows_jigsaw = []
        for name, change_position_1based in allRows_tmpDF.values.tolist():
            labelArr = np.zeros(20, dtype=np.float32)
            label_numeric_aa = 0

            isJigsaw = False
            allRows.append( (name, change_position_1based, label_numeric_aa, labelArr, isJigsaw) )
            isJigsaw = True
            allRows_jigsaw.append( (name, change_position_1based, label_numeric_aa, labelArr, isJigsaw) )

        print("Voxelizing PAI")
        X_eval = self.voxelizeAll(allRows, c, pdbLmdb)
        print("Voxelizing Jigsaw")
        X_eval_jigsaw = self.voxelizeAll(allRows_jigsaw, c, pdbLmdb)

        self.evalName = os.path.basename(variantsDfFilePath).replace(".csv", "")
        
        self.X_eval = X_eval
        self.X_eval_jigsaw = X_eval_jigsaw
        self.c = c
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.allDF = allDF
        self.allRows_tmpDF = allRows_tmpDF

        pdbRepoDict=None


    def voxelizeAll(self, rows, c, pdbLmdb):
        X_chunk, y_chunk, countTuples, sampleWeights, multiz_geneNames, multiz_voxelGridNNs = data_generation_triples(np.array(rows, dtype="object"), c, pdbLmdb, epoch=0, counts=False)

        nFeats = getNFeats( c["nFeatsSeq"],
                            len(c["targetAtoms"]),
                            c["nFeatsEvo"],
                            c["nFeatsAltRef"],
                            c["nFeatsAllAtomDist"],
                            c["nFeatsProtQual"],                    
                            c["includeEvoProfs"],
                            c["includeAlt"],
                            c["includeAllAtomDist"],
                            c["includeProtQual"])

        X_chunk[2] = np.concatenate([np.array([0]), np.cumsum(X_chunk[2])])  #tripleLengths
        X_chunk[5] = np.concatenate([np.array([0]), np.cumsum(X_chunk[5])])  #tripleLengthsGlobal

        print("Restoring voxels")
        X = voxelizeFromTriples(X_chunk[0], 
                                X_chunk[1], 
                                X_chunk[2], 
                                X_chunk[3],
                                X_chunk[4],
                                X_chunk[5],
                                nFeats, 
                                np.float32(c["nVoxels"][0])).astype(np.float32)

        print("Val data shape voxels: %s" % ( str(X.shape) ) )  #str(X_val[1].shape),  %s
        if c["doMultiz"]:
            X = [X,
                 multiz_geneNames,
                 multiz_voxelGridNNs]

        return X


    def performEval(self, model, savePredFile=None):

        t = time.time()

        print("Starting all eval")

        y_pred_eval = model.predict(self.X_eval)
        y_pred_eval_jigsaw = model.predict(self.X_eval_jigsaw)
        y_pred_eval_both = np.mean([y_pred_eval, y_pred_eval_jigsaw], axis=0)

        self.allRows_tmpDF.reset_index(inplace=True, drop=True)
        self.allRows_tmpDF.reset_index(inplace=True, drop=False)

        allDF_merged = self.allDF.merge(self.allRows_tmpDF, on=["name", "change_position_1based"])

        alt_score = y_pred_eval[ allDF_merged["index"].values, allDF_merged.label_numeric_aa_alt.values ]
        alt_score_jigsaw = y_pred_eval_jigsaw[ allDF_merged["index"].values, allDF_merged.label_numeric_aa_alt.values ]
        alt_score_both = y_pred_eval_both[ allDF_merged["index"].values, allDF_merged.label_numeric_aa_alt.values ]

        fullScoreDF = pd.DataFrame({"snp_id": allDF_merged.snp_id, "predPai": alt_score, "predJigsaw": alt_score_jigsaw, "predBoth": alt_score_both})

        print("Done all eval (%s)" % str( time.time() - t ))
        
        return fullScoreDF
