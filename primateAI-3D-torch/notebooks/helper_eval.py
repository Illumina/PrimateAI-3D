import sys
import os
import pickle
import glob
import json
import warnings
import pandas as pd
import numpy as np
import collections

from multiprocessing.pool import Pool
from datetime import datetime

import getpass

from scipy.stats import spearmanr
from scipy.stats import ranksums
from sklearn.metrics import roc_curve, auc, roc_auc_score

aaPosCols = ["gene_name", "change_position_1based", "ref_aa", "alt_aa"]
dnaPosCols = ['chr', 'pos', 'non_flipped_ref', 'non_flipped_alt']
    
def evalClinvar(clinvarFilePath, scoreDF, targetCols, commonVars=True, quiet=True):

    clinvarDF = pd.read_csv(clinvarFilePath)
    clinvarDF.ClinicalSignificance.unique()
    clinvarDF = clinvarDF[clinvarDF.ClinicalSignificance.isin(['Pathogenic', 'Likely pathogenic', 'Benign', 'Likely benign'])]
    clinvarDF = clinvarDF[clinvarDF.ClinStar>=1]
    clinvarDF["ClinicalSignificanceNumeric"] = np.where( clinvarDF.ClinicalSignificance.str.contains("enign"), 0, 1 )
    clinvarDF = clinvarDF[ dnaPosCols + aaPosCols + ["ClinicalSignificanceNumeric"] ].copy()

    mergeCols=None
    if dnaPosCols[-1] in scoreDF.columns:
        if not quiet: print("Using DNA pos cols")
        mergeCols = dnaPosCols
    else:
        if not quiet: print("Using AA pos cols")
        mergeCols = aaPosCols

    if not quiet: print("Total variants with scores before dedup:", len(scoreDF))
    scoreDF = scoreDF[mergeCols + targetCols].drop_duplicates(subset=mergeCols).copy()
    if not quiet: print("Total variants with scores after dedup:", len(scoreDF)) 

    if commonVars:
        if not quiet: print("Total variants with scores before common vars:", len(scoreDF))
        scoreDF = scoreDF.dropna(subset=targetCols)
        if not quiet: print("Total variants with scores after common vars:", len(scoreDF))

    clinvarDF_withAllScores = clinvarDF.merge(scoreDF, on=mergeCols, how="left")

    sigDFs = []
    for targetScoreName in targetCols:

        if not quiet: print("Evaluation variants with scores before no NA %s:" % targetScoreName, len(clinvarDF_withAllScores))
        clinvarDF_withScores_noNA = clinvarDF_withAllScores.dropna(subset=[targetScoreName])
        if not quiet: print("Evaluation variants with scores after no NA %s:" % targetScoreName, len(clinvarDF_withScores_noNA))

        #dddDF_withScores.head()

        sigRows = []
        for (gene_name), grpDF in clinvarDF_withScores_noNA.groupby(["gene_name"]):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fpr, tpr, thresholds = roc_curve(grpDF.ClinicalSignificanceNumeric, grpDF[targetScoreName])
                roc_auc = auc(fpr, tpr)
                nvars = len(grpDF)

            sigRows.append((targetScoreName, nvars, roc_auc, gene_name))

        sigDF = pd.DataFrame(sigRows, columns=["scoreName", "nvars", "roc_auc", "gene_name"])

        sigDF = sigDF[np.isfinite(sigDF.roc_auc)]
        sigDF["statCount"] = 1
        sigDF_grpd = sigDF[["scoreName", "nvars", "roc_auc", "statCount"]].groupby(["scoreName"]).agg({"roc_auc": [np.mean, np.median], "nvars": np.sum, "statCount": np.sum}).reset_index()
        sigDF_grpd.columns = [ "%s_%s" % (s1, s2) if s2 != '' else s1 for s1, s2 in sigDF_grpd.columns.tolist()]
        sigDFs.append(sigDF_grpd)

    sigDF = pd.concat(sigDFs)

    
    sigDF["statName"] = "auc"
    sigDF["stat_mean"] = sigDF["roc_auc_mean"].apply(lambda x: max(x, 1-x))
    sigDF = sigDF[["scoreName", "statName", "stat_mean"]].copy()
    sigDF["dataset"] = "clinvar"
    
    return sigDF

def evalAssays(assayFilePath, scoreDF, targetCols, commonVars=True, quiet=True):

    assayDF = pd.read_csv(assayFilePath)

    assayDF = assayDF[aaPosCols + ["assay_name", "assay_score"]].copy()

    if not quiet: print("Using AA pos cols")
    mergeCols = aaPosCols

    if not quiet: print("rows before dedup:", len(scoreDF))
    scoreDF = scoreDF[mergeCols + targetCols].drop_duplicates(subset=mergeCols).copy()
    if not quiet: print("rows after dedup:", len(scoreDF)) 

    if commonVars:
        if not quiet: print("rows before common vars:", len(scoreDF))
        scoreDF = scoreDF.dropna(subset=targetCols)
        if not quiet: print("rows after common vars:", len(scoreDF))

    assayDF_withAllScores = assayDF.merge(scoreDF, on=mergeCols, how="left")

    sigDFs = []
    for targetScoreName in targetCols:

        if not quiet: print("rows before no NA %s:" % targetScoreName, len(assayDF_withAllScores))
        assayDF_withScores_noNA = assayDF_withAllScores.dropna(subset=[targetScoreName])
        if not quiet: print("rows after no NA %s:" % targetScoreName, len(assayDF_withScores_noNA))

        #dddDF_withScores.head()

        sigRows = []
        for (assay_name, gene_name), grpDF in assayDF_withScores_noNA.groupby(["assay_name", "gene_name"]):

            
            if grpDF[targetScoreName].unique().shape[0] > 5:
                
                corr, sig = spearmanr(grpDF[targetScoreName].astype("float"), grpDF.assay_score.astype("float"))
                r2 = corr * corr

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for statName, stat in [("corr", np.abs(np.float64(corr))),
                                           ("sig", -np.log10(sig)),
                                           ("r2", np.float64(r2))]:
                        sigRows.append((targetScoreName, assay_name, gene_name, statName, stat, len(grpDF.dropna(subset=[targetScoreName]))))

        sigDF = pd.DataFrame(sigRows, columns=["scoreName", "assay_name", "gene_name", "statName", "stat", "nvars"])

        sigDF = sigDF[np.isfinite(sigDF.stat)]
        
        maxDF = sigDF[sigDF.statName == "corr"].groupby(["gene_name"]).agg({"stat": np.max}).reset_index()
        
        maxDF = sigDF.merge(maxDF, on=["gene_name", "stat"])[["gene_name", "assay_name"]].drop_duplicates()
        
        sigDF = sigDF.merge(maxDF, on=["gene_name", "assay_name"])
        
        sigDF["statCount"] = 1
        sigDF_grpd = sigDF[["scoreName", "statName", "stat", "nvars", "statCount"]].groupby(["scoreName", "statName" ]).agg({"stat": [np.mean, np.median], "nvars": np.sum, "statCount": np.sum}).reset_index()
        sigDF_grpd.columns = [ "%s_%s" % (s1, s2) if s2 != '' else s1 for s1, s2 in sigDF_grpd.columns.tolist()]
        
        #sigDFs.append(sigDF)
        
        
        sigDFs.append(sigDF_grpd)

    sigDF = pd.concat(sigDFs)
    
    

    sigDF = sigDF[["scoreName", "statName", "stat_mean"]].copy()
    sigDF = sigDF[sigDF["statName"] == "corr"].copy()
    sigDF["dataset"] = "assay"
    
    return sigDF

def evalDDD(dddFilePath, scoreDF, targetCols, commonVars=True, quiet=True):

    dddDF = pd.read_csv(dddFilePath)
    
    dddDF = dddDF[dnaPosCols + aaPosCols + ["dataset", "label"]].copy()

    dddDF['label_numeric'] = np.where(dddDF.label == "case", 1, 0)

    mergeCols=None
    if dnaPosCols[-1] in scoreDF.columns:
        if not quiet: print("Using DNA pos cols")
        mergeCols = dnaPosCols
    else:
        if not quiet: print("Using AA pos cols")
        mergeCols = aaPosCols

    if not quiet: print("rows before dedup:", len(scoreDF))
    scoreDF = scoreDF[mergeCols + targetCols].drop_duplicates(subset=mergeCols).copy()
    if not quiet: print("rows after dedup:", len(scoreDF)) 
    
    if commonVars:
        if not quiet: print("rows before common vars:", len(scoreDF))
        scoreDF = scoreDF.dropna(subset=targetCols)
        if not quiet: print("rows after common vars:", len(scoreDF))

    dddDF_withAllScores = dddDF.merge(scoreDF, on=mergeCols, how="left")

    sigRows = []
    for targetScoreName in targetCols:

        for dataset, groupDF in dddDF_withAllScores.groupby("dataset"):
            
            groupDF_noNA = groupDF.dropna(subset=[targetScoreName])
            
            sig = ranksums(
                groupDF_noNA[groupDF_noNA.label == "case"][targetScoreName],
                groupDF_noNA[groupDF_noNA.label == "control"][targetScoreName],
            ).pvalue

            warnings.simplefilter("ignore")
            fpr, tpr, thresholds = roc_curve(groupDF_noNA.label_numeric, groupDF_noNA[targetScoreName])
            roc_auc = auc(fpr ,tpr)

            nvars_nonna = len(groupDF_noNA)
            nvars_tot = len(groupDF)
            
            sigRows.append((nvars_tot, nvars_nonna, targetScoreName, dataset, sig, roc_auc))


    sigDF_ddd = pd.DataFrame(
        sigRows, columns=["nvars_tot" ,"nvars_noNA", "scoreName", "dataset", "pval", 'roc_auc']
    )
    
    sigDF_ddd["statName"] = sigDF_ddd["dataset"] + "_log10pval"
    sigDF_ddd["stat_mean"] = -np.log10(sigDF_ddd["pval"])
    
    sigDF_ddd = sigDF_ddd[["scoreName", "statName", "stat_mean"]].copy()
    
    sigDF_ddd["dataset"] = "DDD"

    return sigDF_ddd



def evalUKBB(ukbbFilePath, scoreDF, targetCols, commonVars=True, quiet=True):

    ukbbDF = pd.read_csv(ukbbFilePath)

    ukbbDF = ukbbDF[dnaPosCols + aaPosCols + ["phenotype", "mean_phenotype"]].copy()

    mergeCols=None
    if dnaPosCols[-1] in scoreDF.columns:
        if not quiet: print("Using DNA pos cols")
        mergeCols = dnaPosCols
    else:
        if not quiet: print("Using AA pos cols")
        mergeCols = aaPosCols

    if not quiet: print("rows before dedup:", len(scoreDF))
    scoreDF = scoreDF[mergeCols + targetCols].drop_duplicates(subset=mergeCols).copy()
    if not quiet: print("rows after dedup:", len(scoreDF)) 

    if commonVars:
        if not quiet: print("rows before common vars:", len(scoreDF))
        scoreDF = scoreDF.dropna(subset=targetCols)
        if not quiet: print("rows after common vars:", len(scoreDF))

    ukbbDF_withAllScores = ukbbDF.merge(scoreDF, on=mergeCols, how="left")

    sigDFs = []
    for targetScoreName in targetCols:

        if not quiet: print("rows before no NA %s:" % targetScoreName, len(ukbbDF_withAllScores))
        ukbbDF_withScores_noNA = ukbbDF_withAllScores.dropna(subset=[targetScoreName])
        if not quiet: print("rows after no NA %s:" % targetScoreName, len(ukbbDF_withScores_noNA))

        #dddDF_withScores.head()

        sigRows = []
        for (phenotype, gene_name), grpDF in ukbbDF_withScores_noNA.groupby(["phenotype", "gene_name"]):


            if grpDF[targetScoreName].unique().shape[0] > 5:


                corr, sig = spearmanr(grpDF[targetScoreName].astype("float"), grpDF.mean_phenotype.astype("float"))
                r2 = corr * corr

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for statName, stat in [("corr", np.abs(np.float64(corr))),
                                           ("sig", -np.log10(sig)),
                                           ("r2", np.float64(r2))]:
                        sigRows.append((targetScoreName, phenotype, gene_name, statName, stat, len(grpDF.dropna(subset=[targetScoreName]))))


        sigDF = pd.DataFrame(sigRows, columns=["scoreName", "phenotype", "gene_name", "statName", "stat", "nvars"])

        sigDF = sigDF[np.isfinite(sigDF.stat)]
        sigDF["statCount"] = 1
        sigDF_grpd = sigDF[["scoreName", "statName", "stat", "nvars", "statCount"]].groupby(["scoreName", "statName" ]).agg({"stat": [np.mean, np.median], "nvars": np.sum, "statCount": np.sum}).reset_index()
        sigDF_grpd.columns = [ "%s_%s" % (s1, s2) if s2 != '' else s1 for s1, s2 in sigDF_grpd.columns.tolist()]
        
        #sigDFs.append(sigDF)
        
        sigDFs.append(sigDF_grpd)

    sigDF = pd.concat(sigDFs)
    
    
    
    sigDF = sigDF[["scoreName", "statName", "stat_mean"]].copy()
    sigDF = sigDF[sigDF["statName"] == "corr"].copy()
    
    sigDF["dataset"] = "UKBB"

    return sigDF

