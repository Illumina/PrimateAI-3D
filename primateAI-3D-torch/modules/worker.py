import pandas as pd
import random
import torch
import numpy as np
import argparse
import time
import types
import os, sys

from pathlib import Path

sys.path.insert(1, os.path.join(Path().cwd(), '../'))
sys.path.insert(1, os.path.join(Path().cwd(), '../modules'))

import getpass

from torch.utils.tensorboard import SummaryWriter

from helper.fileOps import mkdir_p

from helper.helper_logging import NnLogger
from helper.helper_input import (
    loadPdbRepo, prepareValSnpDF, getTooLongGenes
)
from helper.helper_data import toPyTorchDict, atomNamesToNum, addJigsawLabels
from helper.helper_evaluation import EvalPositionCollection
from helper.helper_multiz import Multiz
from helper.helper_trainBatching import ProtPossBatchGenerator, ProtPosDataPreproc
from helper.helpers import in_notebook, get_git_revision_hash, get_git_revision_short_hash
from helper.helper_input import readConfig
from helper.helper_sampleWeights import addSampleWeights

from evaluation.evaluation_validation import EvaluationValidation
from evaluation.evaluation_validation_static import EvaluationValidationStatic
from evaluation.evaluation_correlation_multiprot import EvaluationCorrelationMultiprot


from workflow_training import TrainingWorkflow
from loss.loss import PaiLoss
from model_pai3dcnn import PrimateAI3D
from loss.helper_externalScores import combineScores

from helper import globalVars

def set_default_val(config, key, val):
    config[key] = config.get(key, val)

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',504)
pd.set_option('display.width',1000)


if __name__ == "__main__":

    myhost = os.uname()[1]
    print("Host", myhost)

    ####################
    # Parse parameters #
    ####################
    if in_notebook():
        args = types.SimpleNamespace()
        args.datafolder = os.path.join(Path().cwd(), '../') + "/data/package/"
        args.runfolder = "/var/tmp/testrun"
        args.json_conf = "<dataFolder>/ens/0.conf.json"

    if not in_notebook():
        parser = argparse.ArgumentParser()
        parser.add_argument('--json_conf', help="Config json")
        parser.add_argument('--datafolder', help="Folder with PAI-3D input data")
        parser.add_argument('--runfolder', help="Folder for PAI-3D output data")

        args = parser.parse_args()

    os.makedirs( args.runfolder, exist_ok=True )

    ####################
    # Parse config #
    ####################
    configFilePaths = [Path(configFilePath) for configFilePath in args.json_conf.split(',')]
    print("Loading configs %s" % configFilePaths)
    c = readConfig(args.json_conf, args.datafolder, args.runfolder)


    os.environ["OMP_NUM_THREADS"] = str(c["torch_nThreads"])
    os.environ["MKL_NUM_THREADS"] = str(c["torch_nThreads"])
    torch.set_num_threads(c["torch_nThreads"])

    # For some reason this is needed for torch not to enter an endlessloop later with CUDA=11.8
    torch.tensor([1.0]).to(device="cuda")

    if "rank" in c["losses"]:
        if c["loss_rank_misOnly"]:
            c["targetFeatures"].append("prot_isAccByMis")
            c["targetLabels"].append("prot_isAccByMis")
        c["loss_rank_targetLabels"] = ["combinedScore"] if c["loss_rank_combinedScore"] else [s.replace("/", "_") for s in c["loss_rank_externalScores"]]
        c["targetLabels"] += c["loss_rank_targetLabels"]

    if "jigsaw" in c["losses"]:
        c["targetLabels"] += ["label_jigsaw"]

    c["targetLabels"] = list(set(c["targetLabels"]))

    c["torch_device"] = torch.device(c["torch_device"])
    c["revision_hash"] = get_git_revision_hash()
    c["revision_short_hash"] = get_git_revision_short_hash()
    print("Commit used: ", c["revision_short_hash"])
    c["nAtoms"] = len(c["voxel_targetAtoms"])

    ####################
    # Set random seeds #
    ####################
    randomSeed = int(c["randomSeed"])

    np.random.seed(randomSeed)
    torch.manual_seed(randomSeed)
    random.seed(randomSeed)
    torch.cuda.manual_seed(c["randomSeed"])

    ################
    # Init globals #
    ################
    globalVars.init()
    globalVars.mask_value = c["mask_value"]

    ###################
    # Set up tmp dirs #
    ###################
    c["username"] = getpass.getuser()
    c["configName"] = '-'.join([os.path.basename(configFilePath).replace(".conf.json", "") for configFilePath in configFilePaths])
    c["tmpFolder"] = os.path.join(c["tmpBaseFolder"], c["username"], c["configName"])
    mkdir_p(c["tmpFolder"])

    #########################
    # Loading combined data #
    #########################

    pdbRepoFile = c["input_pdbRepoFolder"][0]

    multiz = None
    pdbDict = None

    print("Loading PDB data")
    pdbRepoFile = c["input_pdbRepoFolder"][0]
    pdbDict = loadPdbRepo(c)
    globalVars.tooLongGenes = getTooLongGenes(pdbDict, c)

    #######################
    # Optional components #
    #######################
    if c["input_doMultiz"]:
        multiz = Multiz(c)
        print("Adding multiz tensors to pdbDict")
        multiz.filterAndCombineMultiz(pdbDict)
        multiz.computeNumberAAs(pdbDict)

    ###########################
    # Adding dynamic features #
    ###########################
    print("Adding dynamic features")
    atomNamesToNum(pdbDict, c)

    if "jigsaw" in c["losses"]:
        if not "label_jigsaw" in c["targetLabels"]:
            print("!!! Modifying config: adding 'label_jigsaw' to 'targetLabels'")
            c["targetLabels"].append("label_jigsaw")
        addJigsawLabels(pdbDict, c)

    ###########################
    # Adding dynamic features #
    ###########################
    print("Adding sample weights")
    addSampleWeights(pdbDict, c)

    ###############################
    # Converting to torch tensors #
    ###############################
    print("Converting numpy to pytorch tensors")
    pdbDict = toPyTorchDict(pdbDict)

    if "rank" in c["losses"]:
        if c["loss_rank_combinedScore"]:
            combineScores(pdbDict, c)
    print("Done")

    geneNameToId = {gene_name: i for i, gene_name in enumerate(sorted(pdbDict.keys()))}
    idToGeneName = {i: gene_name for gene_name, i in geneNameToId.items()}

    ####################
    # Batch processing #
    ####################
    print("Preprocessing PDB data")
    dataPreproc = ProtPosDataPreproc(c, pdbDict, geneNameToId, idToGeneName)

    print("Preparing batches")
    dataGenerator_all = ProtPossBatchGenerator("allAugm", c, dataPreproc, pdbDict)
    dataloader_all = torch.utils.data.DataLoader(dataGenerator_all, num_workers=0, batch_size=None, batch_sampler=None)

    dataGenerator_train = ProtPossBatchGenerator("train", c, dataPreproc, pdbDict)
    dataloader_train = torch.utils.data.DataLoader(dataGenerator_train, num_workers=0, batch_size=None, batch_sampler=None)

    dataGenerator_val = ProtPossBatchGenerator("val", c, dataPreproc, pdbDict)
    dataloader_val = torch.utils.data.DataLoader(dataGenerator_val, num_workers=0, batch_size=None, batch_sampler=None)


    ###############
    ### N FEATS ###
    ###############
    multiprot = next(iter(dataGenerator_train))

    subprot = multiprot.protList[0]

    #############
    ### Model ###
    #############
    model = PrimateAI3D(c, pdbDict, multiz=multiz)
    model = model.to(c["torch_device"])
    print(model)

    ############
    ### Loss ###
    ############
    if c["loss_name"] == "sampleWeights_mse":
        lossObj = PaiLoss(c)
    else:
        raise ValueError("Unknown loss: %s" % c["loss_name"])

    ###########################
    ### Counting parameters ###
    ###########################
    modelParas = [paras for namei, paras in model.named_parameters() if not "diff_species_layer" in namei]
    paramsDicts = [{"params": [p for p in modelParas if p.requires_grad]}]

    if "gradnorm" in c["losses"]:
        paramsDicts.append({"params": [p for p in lossObj.parameters() if p.requires_grad], "lr": c["opt_learningRateGradNorm"]})
    if c["input_doMultiz"]:
        multizParas = [paras for namei, paras in model.named_parameters() if "diff_species_layer" in namei]
        assert len(multizParas) > 0
        paramsDicts.append({"params": [p for p in multizParas if p.requires_grad], "lr": c["opt_species_learningRate"]})

    parameter_count = 0
    for dicti in paramsDicts:
        parameter_count += sum([p.numel() for p in dicti["params"]])

    # parameter_count = sum(dict((p.data_ptr(), p.numel()) for p in params).values())
    print(f"Total model parameters: {parameter_count}")

    #################
    ### Optimizer ###
    #################

    if c["opt_optimizer"] == "adam":
        betas = (c['opt_momentum'], 0.999)
        optimizer = torch.optim.Adam(paramsDicts, lr=c["opt_learningRate"], betas=betas)
    elif c["opt_optimizer"] == "sgd":
        optimizer = torch.optim.SGD(paramsDicts, lr=c["opt_learningRate"], momentum=c['opt_momentum'])
    else:
        raise ValueError(f"Unknown optimizer {c['opt_optimizer']}")

    ###############
    ### Logging ###
    ###############
    print("Preparing loggin")
    tboard = SummaryWriter(log_dir=c["runFolder"])
    logger = NnLogger(c, tboard)

    #################################
    ### Validation and Evaluation ###
    #################################

    print("Preparing validation data")
    # Dynamic validation
    eval_val = EvaluationValidation(dataloader_val, lossObj, c, maxProts=-1)
    evalPositionCollection = EvalPositionCollection(c, pdbDict, dataGenerator_all)

    evals = []
    # Static validation
    valSnpFilePath = prepareValSnpDF(c)
    valFilePathTuples = [("pai", valSnpFilePath)]

    eval_staticVal = EvaluationValidationStatic(c, valFilePathTuples, evalPositionCollection)

    print("Preparing evaluation data")

    multiProtCorrEvals = []
    for fileAbs in c["eval_corrMultiProtSnpFilePaths"]:
        evali = EvaluationCorrelationMultiprot(c, fileAbs, evalPositionCollection)
        multiProtCorrEvals.append(evali)

    evals = [eval_staticVal] + multiProtCorrEvals  # , eval_aucMultiprot+ multiProtCorrEvals

    evalPositionCollection.finishCollection()

    ################
    ### Workflow ###
    ################
    print("Generating Wfl")
    wfl = TrainingWorkflow(model, lossObj, optimizer, logger, evals, eval_val,
                           evalPositionCollection)

    ################
    ### Training ###
    ################
    print("Starting training")

    nEpochs = c["maxEpochs"] if "maxEpochs" in c else 10

    s = time.time()
    for epoch in range(nEpochs):
        trainLoss = wfl.trainEpoch(dataloader_train,
                                   epoch,
                                   nBatchesPerTrainEval=c["eval_nBatchesPerTrainEval"],
                                   nEvalsPerEpoch=c["eval_nEvalsPerEpoch"],
                                   saveCheckpoint=True)
        print("Epoch train loss:", trainLoss)

        print("==========")

    print("Training done in %d" % (time.time() - s))