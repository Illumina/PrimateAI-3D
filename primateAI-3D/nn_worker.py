import sys
import json
import random
import os
import shutil
import pandas as pd
import globalVars
import sys
import time
import numpy as np
import keras.losses

from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import model_from_json, load_model
from keras.utils import plot_model
from keras.optimizers import Adam

from nn_worker_helper import loadPdbRepo, loadMultizDB, initRotMatrices, initLabelEncoding, loadAaEncoding, getPaiRows, getJigsawRows, getDsScores, loadDsFile, addScoresToDsDF, convertDsDfToRows
from nn_worker_helper import DataGenerator_triple, getVoxelsForSnpRows_tripleBased_orderSafe

from nn_worker_kerashelper import getLossFunDict, auroc, myacc
from nn_worker_multizhelper import MultizLMDB, calcMultizLabels
from tensorflow import set_random_seed

from eval_ranksum import RanksumEval
from eval_auroc import AurocEval
from nn_worker_multizhelper_tf import MultizLayer
from eval_correlation_multiProt import MultiprotCorrelationEval

mask_value = -1000.0

def prepareTrainSnpDF(c):

    if c["benignTrainFilePath"].endswith(".csv"):
        benignDataFilePath = pd.read_csv(c["benignTrainFilePath"], index_col=0)
    else:
        benignDataFilePath = pd.read_pickle(c["benignTrainFilePath"] )

    if c["pathoTrainFilePath"].endswith(".csv"):
        pathoDataFilePath = pd.read_csv(c["pathoTrainFilePath"], index_col=0)
    else:
        pathoDataFilePath = pd.read_pickle(c["pathoTrainFilePath"]) #, index_col=0)
    
    df = pd.concat([benignDataFilePath, pathoDataFilePath])
    
    if c["enrichWithRefRef"] == "True":
        print("RefRef'ing")
        dfRefRef = df.copy()
        dfRefRef["label_numeric_aa_alt"] = dfRefRef["label_numeric_aa"]
        dfRefRef["altAA"] = dfRefRef["refAA"]
        dfRefRef["non_flipped_alt"] = dfRefRef["non_flipped_ref"]
        dfRefRef["id"] = dfRefRef[ ["chr", "pos", "non_flipped_ref", "non_flipped_alt"] ].astype("str").apply(lambda x: "_".join(x), axis=1)
        dfRefRef[c["targetLabel"]] = 0
        df = pd.concat([df, dfRefRef]).sort_values(["id"]).sample(frac=1)
    
    outputFilePath = os.path.join(c["runFolder"], "trainData.pkl")
    
    df.to_pickle(outputFilePath)

    return outputFilePath

def prepareValSnpDF(c):

    if c["staticValPaiFilePath_benign"].endswith(".csv"):
        benignDataFilePath = pd.read_csv(c["staticValPaiFilePath_benign"], index_col=0)
    else:
        benignDataFilePath = pd.read_pickle(c["staticValPaiFilePath_benign"]) #, index_col=0)

    if c["staticValPaiFilePath_patho"].endswith(".csv"):
        pathoDataFilePath = pd.read_csv(c["staticValPaiFilePath_patho"], index_col=0)
    else:
        pathoDataFilePath = pd.read_pickle(c["staticValPaiFilePath_patho"])  # , index_col=0)
    
    df = pd.concat([benignDataFilePath, pathoDataFilePath])
    outputFilePath = os.path.join(c["runFolder"], "valData.pkl")
    
    df.to_pickle(outputFilePath)

    return outputFilePath

if __name__ == "__main__":

    myhost = os.uname()[1]
    print("Host", myhost)


    configFilePath = sys.argv[1]

    c = json.load(open(configFilePath))

    nSamples = None if int(c["nSamples"]) == -1 else int(c["nSamples"])
    randomSeed = int(c["randomSeed"])
    trainSnpFilePath = prepareTrainSnpDF(c) 

    np.random.seed( randomSeed )
    set_random_seed( randomSeed )
    random.seed( randomSeed )

    globalVars.init()

    outputFolderName = "_".join(os.path.dirname(c["modelOutputFilePath"]).split("/")[-2:])
    c["ramDiskOutputFolderPathBase"] = os.path.join(c["ramdiskDir"], outputFolderName)
    try:
        print("Deleting", c["ramDiskOutputFolderPathBase"])
        shutil.rmtree(c["ramDiskOutputFolderPathBase"])
    except:
        print("Not found, so not deleting: ", c["ramDiskOutputFolderPathBase"])

    c["scratchOutputFolderPathBase"] = os.path.join("/scratch/thamp", outputFolderName)

    print("Loading PDB repo")
    pdbLmdb, pdbLmdb_path, mins, maxs, geneNameToId, idToGeneName = loadPdbRepo(c["pdbRepoFiles"], c["evoFeatsSel"], c["scratchOutputFolderPathBase"])
    globalVars.globalVars["pdbLmdb"] = pdbLmdb
    globalVars.globalVars["geneNameToId"] = geneNameToId
    globalVars.globalVars["idToGeneName"] = idToGeneName

    print("Opening Multiz DB")
    multizLmdb, multizLmdb_path = loadMultizDB(c, None)
    globalVars.globalVars["multizLmdb"] = multizLmdb

    if c["doMultiz"]:
        multizLmdbWrapper = MultizLMDB(multizLmdb, c)
        globalVars.globalVars["multizLmdbWrapper"] = multizLmdbWrapper

    dsDF = None
    if c["reduceToVarsWithDS"]:
        print("Loading full DS DF")
        dsDF = pd.read_pickle(c["dsPklFilePath"])
        dsDF["name"] = dsDF["gene_name"]
        dsDF = dsDF[["name", "change_position_1based"]].drop_duplicates().copy()


    paiRows_val = []
    jigsawRows_val = []
    paiRows_train = []
    jigsawRows_train = []
    dsRows_val = []
    dsRows_train = []

    trainRowsList = []
    if c["doDS"]:
        print("Loading DS rows")

        print("Loading all DS related scores")


        dsScoresDF = getDsScores(c["dsScoresFilePath"], c["dsRankCols"], c["dsNormMethod"], c["dsDoQuantNorm"])

        print("Done")
        
        if c["dsDoJigsawSample"]:
            dsSampleDF_train = loadDsFile(c["jigsawTrainFilePath"], targetColumns=[])
            dsSampleDF_train["probs_patho"] = 0.0
        else:
            dsSampleDF_train = loadDsFile(c["dsTrainFilePath"], targetColumns=["probs_patho"])
        dsSampleDF_train = addScoresToDsDF(dsSampleDF_train, dsScoresDF)
        dsRows_train = convertDsDfToRows(dsSampleDF_train, "dsScore", c["binaryDsLabels"], constantToAdd=10, maxSamples=c["maxSnpSamples"])

        random.shuffle(dsRows_train)
        trainRowsList.append( dsRows_train )

        print("Train rows DS", len(dsRows_train))

        if c["dsDoJigsawSample"]:
            dsSampleDF_val = loadDsFile(c["staticValJigsawFilePath"], targetColumns=[])
            dsSampleDF_val["probs_patho"] = 0.0
        else:
            dsSampleDF_val = loadDsFile(c["staticValDsFilePath"], targetColumns=["probs_patho"])
        dsSampleDF_val = addScoresToDsDF(dsSampleDF_val, dsScoresDF)
        dsRows_val = convertDsDfToRows(dsSampleDF_val, "dsScore", c["binaryDsLabels"], constantToAdd=10, maxSamples=c["maxSnpSamples"])

    if c["doJigsaw"]:
        print("Loading Jigsaw rows")
        jigsawRows_val = getJigsawRows(c["staticValJigsawFilePath"], pdbLmdb, dsDF, paiFormat=c["jigsawInPaiFormat"], maxSamples=c["maxSnpSamples"])
        
        if c["doMultiz"]:
            jigsawRows_val = calcMultizLabels(jigsawRows_val, multizLmdbWrapper, nProcs=1)

        jigsawRows_train = getJigsawRows(c["jigsawTrainFilePath"], pdbLmdb, dsDF, paiFormat=c["jigsawInPaiFormat"], maxSamples=c["maxSnpSamples"])

        if c["doMultiz"]:
            print("Conversion start")
            t = time.time()
            jigsawRows_train = calcMultizLabels(jigsawRows_train, multizLmdbWrapper, nProcs=10)
            print("Conversion end", time.time() - t)
        random.shuffle(jigsawRows_train)
        trainRowsList.append(jigsawRows_train)

        print("Train rows Jigsaw", len(jigsawRows_train))

    if c["doPai"]:
        paiRows_val = getPaiRows(prepareValSnpDF(c), dsDF, maxSamples=c["maxSnpSamples"])

        paiRows_train = getPaiRows(trainSnpFilePath, dsDF, maxSamples=c["maxSnpSamples"])
        random.shuffle(paiRows_train)
        trainRowsList.append( paiRows_train )

        print("Train rows Pai", len(paiRows_train))

    if "nTrainSamples" in c:
        targetTrainRows = c["nTrainSamples"]
    else:
        targetTrainRows = min([len(l) for l in trainRowsList])
    print("Target n rows", targetTrainRows)

    if not c["jigsawInPaiFormat"]:
        raise Exception("Not implemented")

    trainRows = []
    for l in trainRowsList:
        trainRows.extend(l[:targetTrainRows])

    print("Val row counts ", len(paiRows_val), len(jigsawRows_val), len(dsRows_val))

    valRows = paiRows_val + jigsawRows_val


    random.shuffle(valRows)

    print("Combined train row count ", len(trainRows))
    print("Combined val row count ", len(valRows))

    try:
        del paiRows_val
        del jigsawRows_val
        del paiRows_train
        del jigsawRows_train
        del dsRows_val
        del dsRows_train
    except:
        print("Error deleting!")

    print("Initializing label encoding")
    labelsDict, evoLabelsDict = initLabelEncoding(c, pdbLmdb)
    globalVars.globalVars["labelsDict"] = labelsDict
    del evoLabelsDict

    aaEncodingDict = loadAaEncoding()
    globalVars.globalVars["aaEncodingDict"] = aaEncodingDict

    globalVars.globalVars["rotMatrices"] = initRotMatrices(c)

    print("Generating val data")

    print(valRows[:10])
    print(valRows[-10:])
    
    X_val, y_val, countTuples, sampleWeights_val = getVoxelsForSnpRows_tripleBased_orderSafe(valRows, c, pdbLmdb)
    print("Done")
    print("Val sample weights: ", sampleWeights_val[:1000])

    if isinstance(X_val, list):
        for Xval_i in X_val:
            print("All values finite: ", np.isfinite(Xval_i).all())
    else:
        print("All values finite: ", np.isfinite(X_val).all())

    countDF = pd.DataFrame(countTuples, columns=["geneName", "change_position_1_based", "aaNumRef", "aaNumAlt", "jigsaw", "countsBoxAll", "countsBoxUsed"])
    print(countDF.head())
    print(countDF[["countsBoxAll", "countsBoxUsed"]].mean())
    countDFOutputFilePath = os.path.join(c["runFolder"], "countDF.csv")
    countDF.to_csv(countDFOutputFilePath)
    print("Mean")
    print(countDF[["countsBoxAll", "countsBoxUsed"]].mean())
    print("Min")
    print(countDF[["countsBoxAll", "countsBoxUsed"]].min())
    print("Max")
    print(countDF[["countsBoxAll", "countsBoxUsed"]].max())

    callbacks = []

    if c["ranksum_eval"]:
        ranksumEvalFiles= c["evalRanksumSnpFilePaths"]
        for ranksumEvalFile in ranksumEvalFiles:
            print("Adding Ranksum callback %s" % ranksumEvalFile)
            ranksumEval =  RanksumEval(c, ranksumEvalFile, pdbLmdb, dsDF)
            callbacks.append(ranksumEval)

    corrEvalFiles = c["evalCorrMultiProtSnpFilePaths"]
    for corrEvalFile in corrEvalFiles:
        print("Adding Multiprot correlation callback %s" % corrEvalFile)
        callbacks.append( MultiprotCorrelationEval(c, corrEvalFile, pdbLmdb, dsDF) )
    callbacks.append( MultiprotCorrelationEval(c, c["staticValDsFilePath"], pdbLmdb, dsDF) )

    if c["auroc_eval"]:
        print("Adding Auroc callback")
        callbacks.append( AurocEval( c, X_val, y_val, valRows ) )

    print("Starting NN")

    lossFunDict = getLossFunDict(c)

    keras.losses.masked_loss_function = lossFunDict["masked_loss_function"]
    keras.losses.masked_binary_crossentropy = lossFunDict["masked_binary_crossentropy"]
    keras.losses.masked_mean_squared_error = lossFunDict["masked_mean_squared_error"]
    keras.losses.lossMSEPairwise = lossFunDict["lossMSEPairwise"]
    keras.losses.lossLogisticPairwise = lossFunDict["lossLogisticPairwise"]

    keras.metrics.auroc = auroc
    keras.metrics.myacc = myacc
    
    if os.path.exists(c["modelOutputFilePath"]):
        print("Loading from existing model")
        modeli = load_model(c["modelOutputFilePath"])
    else:
        fromJson = False
        if "origModelPath" in c:
            if c["origModelPath"] is None:
                fromJson=True
            else:
                print("Transfering model")
                modeli = load_model(c["origModelPath"])
        else:
            fromJson = True

        if fromJson:
            print("New model")
            modeli = model_from_json(open(c["modelInputFilePath"]).read(), {'MultizLayer': MultizLayer})
    
    plot_model(modeli, to_file=c["archPlotFilePath"], show_shapes=True, show_layer_names=True)

    print("Test samples: %d" % len(valRows))

    myTrainDG = DataGenerator_triple(modeli, c, "train", trainRows, pdbLmdb)

    tensorboard_callback = TensorBoard(log_dir=c["runFolder"])

    keras_modelcheckpoint_valloss = ModelCheckpoint(c["modelOutputFilePath"],
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='min',
                                            period=1)

    keras_modelcheckpoint_valauc = ModelCheckpoint(c["modelOutputFilePath"] + ".auc",
                                            monitor='val_auroc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max',
                                            period=1)

    keras_modelcheckpoint_jigsawauc = ModelCheckpoint(c["modelOutputFilePath"] + ".maxPai.auc",
                                            monitor='auroc_jigsaw',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max',
                                            period=1)

    keras_modelcheckpoint_paiauc = ModelCheckpoint(c["modelOutputFilePath"] + ".maxJig.auc",
                                            monitor='auroc_pai',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max',
                                            period=1)


    keras_modelcheckpoint_meanauc = ModelCheckpoint(c["modelOutputFilePath"] + ".maxMean.auc",
                                            monitor='auroc_mean',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max',
                                            period=1)

    keras_earlystopping = EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=50,
                                        verbose=0,
                                        mode='min',
                                        baseline=None)

    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  mode='min',
                                  factor=0.1,
                                  verbose=1,
                                  patience=15,
                                  min_lr=1e-20)

    keras_csvlogger = CSVLogger(c["logCsvFilePath"], separator=',', append=False)

    keras_adam = Adam(lr=c["learningRate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    modeli.compile(loss=lossFunDict[c["lossFun"]],
                   optimizer=keras_adam,
                   metrics=['accuracy', auroc, myacc])

    callbacks.extend([
                    tensorboard_callback,
                    keras_modelcheckpoint_valloss,
                    keras_modelcheckpoint_valauc,
                    keras_modelcheckpoint_paiauc,
                    keras_modelcheckpoint_meanauc,
                    keras_earlystopping,
                    keras_csvlogger])

    print(c["nDataGenWorkers"])
    print("Multi:",  c["nDataGenWorkers"] > 1)

    modeli.fit_generator(generator=myTrainDG,
                         validation_data= (X_val, y_val, np.array(sampleWeights_val)), #myTestDG, #,
                         use_multiprocessing=c["nDataGenWorkers"] > 1,
                         workers=c["nDataGenWorkers"],
                         verbose=True,
                         epochs=c["maxEpochs"] if "maxEpochs" in c else 100,
                         max_queue_size=c["maxQueueSize"],
                         callbacks=callbacks)

    myTrainDG.cancelToken.value = True
    myTrainDG.voxelizationProcess.join()

    time.sleep(5)
    print("Removing", c["ramDiskOutputFolderPathBase"])
    shutil.rmtree(c["ramDiskOutputFolderPathBase"])
    print("Removing", c["scratchOutputFolderPathBase"])
    shutil.rmtree(c["scratchOutputFolderPathBase"])

    shutil.rmtree(multizLmdb_path)