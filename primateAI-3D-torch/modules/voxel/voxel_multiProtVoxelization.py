# This module voxelizes all variants in a multiprot in one operation
# entry point is the "voxelize" function; all other functions are helper functions
# Output is
# (a) a distance profile containing distances of each voxel center to the nearest amino acid type and atom type
# (b) mapping from each voxel to the nearest residue in sequence, represented by its residueId/change_position_1based

import torch
from torch_scatter import scatter_max

from voxel.voxel_grid import Voxelgrid
from voxel.voxel_subProtVoxelization import getVoxelDict, getCopyTensor
from helper.helper_voxel import rep_1d, rep_3d


def voxelize(multiprot, grid, c):
    protList = multiprot.protList

    features_no_multiz = [f for f in c['model_voxels_extraVoxelFeatures'] if not f in c['input_featsDiffMultiz']]

    atom_changePos, \
    atom_coords, \
    atom_resid, \
    atom_atomNameNum, \
    atom_resnamenum, \
    atom_batchResIds0Based, \
    atom_centralCalphaCoords, \
    atom_varNr, \
    atom_isTargetResidue, \
    atom_idx, \
    prot_feats0based, \
    prot_multizAlis0based, \
    vars_changePoss,\
    nAtomsPerVar = initFromProtList(
        protList,
        features_no_multiz,
        c['input_featsDiffMultiz'],
        c["torch_device"],
    )


    atom_box_coords, \
    atom_box_isTargetResidue, \
    atom_box_idx, \
    atom_box_atomCenterCoords = centerRotateBox(atom_coords,
                                                            atom_isTargetResidue,
                                                            atom_centralCalphaCoords,
                                                            atom_idx,
                                                            grid,
                                                            vars_changePoss,
                                                            nAtomsPerVar)

    #printMemUsage("3")

    atomCenterPair_sel_centerCoordsFlat, \
    atomCenterPair_sel_vics, \
    atomCenterPair_sel_atomIdx = atomCenterPairPrep(atom_box_atomCenterCoords,
                       atom_box_coords,
                       atom_box_isTargetResidue,
                       atom_box_idx,
                       grid)

    #printMemUsage("4")

    atomCenterPair_sel_isTargetResidue = atom_isTargetResidue[ atomCenterPair_sel_atomIdx ]

    #printMemUsage("5")

    voxels_proteinBatch_allAtom_jigsaw, \
    voxels_proteinBatch_perAA_jigsaw, \
    voxels_proteinBatch_allAtom_full, \
    voxels_proteinBatch_perAA_full, \
    voxels_proteinBatch_batchResIds0Based_jigsaw, \
    voxels_proteinBatch_batchResIds0Based_full = allocateOutput(grid,
                                                               c,
                                                               vars_changePoss)

    #printMemUsage("6")

    #The first voxelization (jigsawEnvWfl) happens with the target residue removed,
    # the second (jigsawResWfl) is just a delta and only looks at the atoms of the residue that was removed.
    # This is for efficiency reasons.
    voxels_argmax_centerIdx_beforeInSequence_jigsaw,\
    voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw, \
    voxels_argmax_centerIdx_jigsaw, \
    voxels_argmax_batchResIds0Based_jigsaw  =  jigsawEnvWfl(atomCenterPair_sel_atomIdx,
                 atomCenterPair_sel_isTargetResidue,
                 atom_varNr,
                 atom_changePos,
                 atom_resid,
                 atom_batchResIds0Based,
                 atom_atomNameNum,
                 atom_resnamenum,
                 atomCenterPair_sel_centerCoordsFlat,
                 atomCenterPair_sel_vics,
                 grid,
                 c,
                 voxels_proteinBatch_allAtom_jigsaw,
                 voxels_proteinBatch_perAA_jigsaw,
                 voxels_proteinBatch_batchResIds0Based_jigsaw)


    #printMemUsage("7")

    voxels_proteinBatch_allAtom_full[:] = voxels_proteinBatch_allAtom_jigsaw[:]
    voxels_proteinBatch_perAA_full[:] = voxels_proteinBatch_perAA_jigsaw[:]
    voxels_proteinBatch_batchResIds0Based_full[:] = voxels_proteinBatch_batchResIds0Based_jigsaw[:]

    #printMemUsage("8")
    
    #The first voxelization (jigsawEnvWfl) happens with the target residue removed,
    # the second (jigsawResWfl) is just a delta and only looks at the atoms of the residue that was removed.
    # This is for efficiency reasons.
    voxels_argmax_centerIdx_full,\
    voxels_argmax_centerIdx_perAA_full,\
    voxels_argmax_batchResIds0Based_full = jigsawResWfl(atomCenterPair_sel_atomIdx,
                 atomCenterPair_sel_isTargetResidue,
                 atom_varNr,
                 atom_changePos,
                 atom_resid,
                 atom_batchResIds0Based,
                 atom_atomNameNum,
                 atom_resnamenum,
                 atomCenterPair_sel_centerCoordsFlat,
                 atomCenterPair_sel_vics,
                 grid,
                 c,
                 voxels_proteinBatch_allAtom_full,
                 voxels_proteinBatch_perAA_full,
                 voxels_proteinBatch_batchResIds0Based_full)

    #printMemUsage("9")

    updateVicSigns(voxels_proteinBatch_allAtom_jigsaw,
                   voxels_proteinBatch_perAA_jigsaw,
                   voxels_argmax_centerIdx_beforeInSequence_jigsaw,
                   voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw,
                   voxels_proteinBatch_allAtom_full,
                   voxels_proteinBatch_perAA_full,
                   voxels_argmax_centerIdx_full,
                   voxels_argmax_centerIdx_perAA_full)

    #printMemUsage("10")

    return voxels_proteinBatch_allAtom_jigsaw, \
           voxels_proteinBatch_perAA_jigsaw, \
           voxels_argmax_centerIdx_jigsaw, \
           voxels_argmax_batchResIds0Based_jigsaw, \
           voxels_proteinBatch_allAtom_full, \
           voxels_proteinBatch_perAA_full, \
           voxels_argmax_centerIdx_full, \
           voxels_argmax_batchResIds0Based_full, \
           prot_feats0based, \
           prot_multizAlis0based, \

def concatListDict(listDict):
    # assume elements of listDict all have the same keys
    keys = listDict[0].keys()
    output_dict = {}
    for key in keys:
        concat = torch.concat([dict_[key] for dict_ in listDict])
        output_dict[key] = concat
    return output_dict


def initFromProtList(protList, featureList, featureMultizList, device):
    list_prot_nAtoms = []
    list_prot_nVars = []
    list_prot_feats0based = []
    list_prot_multizAlis0based = []

    list_vars_atomCopies = []
    list_vars_changePoss = []
    list_vars_calphaCoords = []
    list_vars_atomResIds = []
    list_vars_atomAtomNameNum = []
    list_vars_atomResNameNum = []
    list_vars_batchResIds0Based = []

    # protNr = []

    residueOffset = 0
    for prot in protList:
        voxelDict = getVoxelDict(prot, device)

        list_prot_nAtoms.append(voxelDict["atom_coords"].shape[0])


        nvars = prot.changePoss.shape[0]
        list_prot_nVars.append(nvars)

        allFeats0based = prot.getCombinedFeatureTensor_0based(featureList, device)
        list_prot_feats0based.append( allFeats0based )
        if len(featureMultizList):
            allMultizFeats0b = prot.getDictTensor_0based(featureMultizList, device)
            list_prot_multizAlis0based.append(allMultizFeats0b)

        vars_atom_coords_flat, \
            vars_atom_resid, \
            vars_atom_atomNamesNum, \
            vars_atom_resnamenum, \
            vars_atom_batchResIds0Based = getCopyTensor(voxelDict, nvars, residueOffset)

        list_vars_atomCopies.append(vars_atom_coords_flat)

        list_vars_atomResIds.append(vars_atom_resid)

        list_vars_atomAtomNameNum.append(vars_atom_atomNamesNum)

        list_vars_atomResNameNum.append(vars_atom_resnamenum)

        list_vars_batchResIds0Based.append(vars_atom_batchResIds0Based)

        list_vars_changePoss.append(prot.changePoss)

        list_vars_calphaCoords.append(prot.centralCaAtomCoords)

        residueOffset += allFeats0based.shape[0]

    prot_nAtoms = torch.tensor(list_prot_nAtoms, device=device)
    prot_nVars = torch.tensor(list_prot_nVars, device=device)

    prot_feats0based = torch.concat(list_prot_feats0based)

    if len(featureMultizList):
        prot_multizAlis0based = concatListDict(list_prot_multizAlis0based)
    else:
        prot_multizAlis0based = None

    vars_changePoss = torch.concat(list_vars_changePoss).to(device=device)
    vars_calphaCoords = torch.concat(list_vars_calphaCoords).to(device=device)

    # result: vector os length #variants, contains number of atoms of corresponding protein
    nAtomsPerVar = prot_nAtoms.repeat_interleave(prot_nVars)
    vars_varNr = torch.arange(0, vars_changePoss.shape[0], device=device)  # self.prot_protNr.repeat_interleave(self.prot_nVars)

    # change pos of each variant expanded to match repeated atoms
    atom_changePos = vars_changePoss.repeat_interleave(nAtomsPerVar)

    atom_coords = torch.concat(list_vars_atomCopies).to(device=device)
    atom_resid = torch.concat(list_vars_atomResIds).to(device=device)
    atom_atomNameNum = torch.concat(list_vars_atomAtomNameNum).to(device=device)
    atom_resnamenum = torch.concat(list_vars_atomResNameNum).to(device=device)
    atom_batchResIds0Based = torch.concat(list_vars_batchResIds0Based).to(device=device)

    atom_centralCalphaCoords = vars_calphaCoords.repeat_interleave(nAtomsPerVar, dim=0)

    atom_varNr = vars_varNr.repeat_interleave(nAtomsPerVar, dim=0)

    atom_isTargetResidue = atom_resid == atom_changePos

    atom_idx = torch.arange(0, atom_coords.shape[0], device=device)

    return atom_changePos,\
           atom_coords,\
           atom_resid,\
           atom_atomNameNum,\
           atom_resnamenum, \
           atom_batchResIds0Based,\
           atom_centralCalphaCoords,\
           atom_varNr,\
           atom_isTargetResidue,\
           atom_idx, \
           prot_feats0based, \
           prot_multizAlis0based, \
           vars_changePoss, \
           nAtomsPerVar



def centerRotateBox(atom_coords,
                    atom_isTargetResidue,
                    atom_centralCalphaCoords,
                    atom_idx,
                    grid,
                    vars_changePoss,
                    nAtomsPerVar):



    #Rotate atom coordinates
    atom_coords_centered = rotate(atom_coords,
                                   atom_centralCalphaCoords,
                                   grid,
                                   vars_changePoss,
                                   nAtomsPerVar)

    #reduce atoms to box
    atom_box_coords, \
        atom_box_isTargetResidue,\
        atom_box_idx = reduceToBox(atom_coords_centered,
                                                            grid,
                                                            atom_isTargetResidue,
                                                            atom_idx)

    # Get nearest voxel center for each atom in box
    atom_box_atomCenterCoords = Voxelgrid.getNearestBoxCenter(atom_box_coords,
                                                              grid.voxelSize,
                                                              grid.maxCenterCoord)

    return atom_box_coords, \
           atom_box_isTargetResidue, \
           atom_box_idx, \
           atom_box_atomCenterCoords



def atomCenterPairPrep(atom_box_atomCenterCoords,
                       atom_box_coords,
                       atom_box_isTargetResidue,
                       atom_box_idx,
                       grid):

    #printMemUsage("3-1")
    atomCenterPair_centerCoordsFlat, \
    atomCenterPair_vics, \
    atomCenterPair_goodIdx_bool, \
    atomCenterPair_atomIdx = generateAtomCenterPairs(atom_box_atomCenterCoords,
                                                                    grid,
                                                                    atom_box_coords,
                                                                    atom_box_isTargetResidue,
                                                                    atom_box_idx)

    #printMemUsage("3-2")
    atomCenterPair_sel_centerCoordsFlat, \
    atomCenterPair_sel_vics, \
    atomCenterPair_sel_atomIdx = reduceToGood(atomCenterPair_goodIdx_bool,
                                                             atomCenterPair_centerCoordsFlat,
                                                             atomCenterPair_vics,
                                                             atomCenterPair_atomIdx,
                                                             grid)
    #printMemUsage("3-3")

    return atomCenterPair_sel_centerCoordsFlat, \
           atomCenterPair_sel_vics, \
           atomCenterPair_sel_atomIdx


def jigsawEnvWfl(atomCenterPair_sel_atomIdx,
                    atomCenterPair_sel_isTargetResidue,
                    atom_varNr,
                    atom_changePos,
                    atom_resid,
                    atom_batchResIds0Based,
                    atom_atomNameNum,
                    atom_resnamenum,
                    atomCenterPair_sel_centerCoordsFlat,
                    atomCenterPair_sel_vics,
                    grid,
                    c,
                    voxels_proteinBatch_allAtom_jigsaw,
                    voxels_proteinBatch_perAA_jigsaw,
                    voxels_proteinBatch_batchResIds0Based_jigsaw):

    #printMemUsage("6-1")
    atomCenterPair_selJigsawEnv_atomIdx, \
    atomCenterPair_selJigsawEnv_varNr, \
    atomCenterPair_selJigsawEnv_atomNameNum, \
    atomCenterPair_selJigsawEnv_resNameNum, \
    atomCenterPair_selJigsawEnv_centerCoordsFlat,\
    atomCenterPair_selJigsawEnv_vics = reduceByIdx(atomCenterPair_sel_atomIdx,
                                                                ~atomCenterPair_sel_isTargetResidue,
                                                                atom_varNr,
                                                                atom_atomNameNum,
                                                                atom_resnamenum,
                                                                atomCenterPair_sel_centerCoordsFlat,
                                                                atomCenterPair_sel_vics)

    #printMemUsage("6-2")

    voxels_proteinBatch_allAtom_jigsaw_argmax = voxelize_scatter_allAtom(grid,
                                       voxels_proteinBatch_allAtom_jigsaw,
                                       atomCenterPair_selJigsawEnv_centerCoordsFlat,
                                       atomCenterPair_selJigsawEnv_varNr,
                                       atomCenterPair_selJigsawEnv_vics)

    #printMemUsage("6-3")

    voxels_proteinBatch_perAA_jigsaw_argmax,\
    atomCenterPair_selJigsawEnv_atomIdx_targeted = voxelize_scatter_perAA(grid,
                                   c,
                                   voxels_proteinBatch_perAA_jigsaw,
                                   atomCenterPair_selJigsawEnv_atomIdx,
                                   atomCenterPair_selJigsawEnv_centerCoordsFlat,
                                   atomCenterPair_selJigsawEnv_varNr,
                                   atomCenterPair_selJigsawEnv_vics,
                                   atomCenterPair_selJigsawEnv_atomNameNum,
                                   atomCenterPair_selJigsawEnv_resNameNum)

    #printMemUsage("6-4")

    #TODO do we need changeposs?
    voxels_argmax_centerIdx_beforeInSequence_jigsaw, \
    voxels_argmax_centerIdx_jigsaw, \
    voxels_argmax_residueIds_jigsaw,\
    voxels_argmax_batchResIds0Based_jigsaw = getVoxelResIds(voxels_proteinBatch_allAtom_jigsaw_argmax,
                                                               atomCenterPair_selJigsawEnv_atomIdx,
                                                               atom_resid,
                                                               atom_batchResIds0Based,
                                                               atom_changePos)
    #printMemUsage("6-5")

    voxels_proteinBatch_batchResIds0Based_jigsaw.ravel()[voxels_argmax_centerIdx_jigsaw] = voxels_argmax_batchResIds0Based_jigsaw

    #printMemUsage("6-6")


    #TODO do we need anything except beforeInSequence here?
    voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw, \
    voxels_argmax_centerIdx_perAA_jigsaw, \
    voxels_argmax_residueIds_perAA_jigsaw, \
    voxels_argmax_changePoss_perAA_jigsaw = getVoxelResIds_perAA(voxels_proteinBatch_perAA_jigsaw_argmax,
                                                                           atomCenterPair_selJigsawEnv_atomIdx_targeted,
                                                                           atom_resid,
                                                                           atom_changePos)

    #printMemUsage("6-7")

    return  voxels_argmax_centerIdx_beforeInSequence_jigsaw,\
            voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw, \
            voxels_argmax_centerIdx_jigsaw, \
            voxels_argmax_batchResIds0Based_jigsaw


def jigsawResWfl(atomCenterPair_sel_atomIdx,
                 atomCenterPair_sel_isTargetResidue,
                 atom_varNr,
                 atom_changePos,
                 atom_resid,
                 atom_batchResIds0Based,
                 atom_atomNameNum,
                 atom_resnamenum,
                 atomCenterPair_sel_centerCoordsFlat,
                 atomCenterPair_sel_vics,
                 grid,
                 c,
                 voxels_proteinBatch_allAtom_full,
                 voxels_proteinBatch_perAA_full,
                 voxels_proteinBatch_batchResIds0Based_full):


    atomCenterPair_selJigsawRes_atomIdx, \
    atomCenterPair_selJigsawRes_varNr, \
    atomCenterPair_selJigsawRes_atomNameNum, \
    atomCenterPair_selJigsawRes_resNameNum, \
    atomCenterPair_selJigsawRes_centerCoordsFlat,\
    atomCenterPair_selJigsawRes_vics = reduceByIdx(atomCenterPair_sel_atomIdx,
                                                             atomCenterPair_sel_isTargetResidue,
                                                             atom_varNr,
                                                             atom_atomNameNum,
                                                             atom_resnamenum,
                                                             atomCenterPair_sel_centerCoordsFlat,
                                                             atomCenterPair_sel_vics)


    voxels_proteinBatch_allAtom_full_argmax = voxelize_scatter_allAtom(grid,
                                                                                   voxels_proteinBatch_allAtom_full,
                                                                                   atomCenterPair_selJigsawRes_centerCoordsFlat,
                                                                                   atomCenterPair_selJigsawRes_varNr,
                                                                                   atomCenterPair_selJigsawRes_vics)


    voxels_proteinBatch_perAA_full_argmax,\
    atomCenterPair_selJigsawRes_atomIdx_targeted = voxelize_scatter_perAA(grid,
                                                                           c,
                                                                           voxels_proteinBatch_perAA_full,
                                                                           atomCenterPair_selJigsawRes_atomIdx,
                                                                           atomCenterPair_selJigsawRes_centerCoordsFlat,
                                                                           atomCenterPair_selJigsawRes_varNr,
                                                                           atomCenterPair_selJigsawRes_vics,
                                                                           atomCenterPair_selJigsawRes_atomNameNum,
                                                                           atomCenterPair_selJigsawRes_resNameNum)


    voxels_argmax_centerIdx_beforeInSequence_full, \
    voxels_argmax_centerIdx_full, \
    voxels_argmax_residueIds_full,\
    voxels_argmax_batchResIds0Based_full = getVoxelResIds(voxels_proteinBatch_allAtom_full_argmax,
                                                               atomCenterPair_selJigsawRes_atomIdx,
                                                               atom_resid,
                                                               atom_batchResIds0Based,\
                                                               atom_changePos)
    voxels_proteinBatch_batchResIds0Based_full.ravel()[voxels_argmax_centerIdx_full] = voxels_argmax_batchResIds0Based_full

    # #TODO do we need anything except beforeInSequence here?
    voxels_argmax_centerIdx_beforeInSequence_perAA_full, \
    voxels_argmax_centerIdx_perAA_full, \
    voxels_argmax_residueIds_perAA_full, \
    voxels_argmax_changePoss_perAA_full = getVoxelResIds_perAA(voxels_proteinBatch_perAA_full_argmax,
                                                                         atomCenterPair_selJigsawRes_atomIdx_targeted,
                                                                         atom_resid,
                                                                         atom_changePos)

    return  voxels_argmax_centerIdx_full,\
            voxels_argmax_centerIdx_perAA_full,\
            voxels_argmax_batchResIds0Based_full


def updateVicSigns(voxels_proteinBatch_allAtom_jigsaw,
                   voxels_proteinBatch_perAA_jigsaw,
                   voxels_argmax_centerIdx_beforeInSequence_jigsaw,
                   voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw,
                   voxels_proteinBatch_allAtom_full,
                   voxels_proteinBatch_perAA_full,
                   voxels_argmax_centerIdx_full,
                   voxels_argmax_centerIdx_perAA_full):

    #UPDATE VICINITY signs (before after target residue in sequence)
    voxels_proteinBatch_allAtom_jigsaw.ravel()[voxels_argmax_centerIdx_beforeInSequence_jigsaw] = voxels_proteinBatch_allAtom_jigsaw.ravel()[voxels_argmax_centerIdx_beforeInSequence_jigsaw] * -1
    voxels_proteinBatch_perAA_jigsaw.ravel()[voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw] = voxels_proteinBatch_perAA_jigsaw.ravel()[voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw] * -1


    voxels_proteinBatch_allAtom_full.ravel()[voxels_argmax_centerIdx_beforeInSequence_jigsaw] = voxels_proteinBatch_allAtom_full.ravel()[voxels_argmax_centerIdx_beforeInSequence_jigsaw] * -1
    voxels_proteinBatch_allAtom_full.ravel()[voxels_argmax_centerIdx_full] = torch.abs(voxels_proteinBatch_allAtom_full.ravel()[voxels_argmax_centerIdx_full])


    voxels_proteinBatch_perAA_full.ravel()[voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw] = voxels_proteinBatch_perAA_full.ravel()[voxels_argmax_centerIdx_beforeInSequence_perAA_jigsaw] * -1
    voxels_proteinBatch_perAA_full.ravel()[voxels_argmax_centerIdx_perAA_full] = torch.abs(voxels_proteinBatch_perAA_full.ravel()[voxels_argmax_centerIdx_perAA_full])


def reduceByIdx(atomCenterPair_atomIdx,
                atomCenterPair_targeted,
                atom_varNr,
                atom_atomNameNum,
                atom_resnamenum,
                atomCenterPair_centerCoordsFlat,
                atomCenterPair_vics):

    atomCenterPair_atomIdx_targeted = atomCenterPair_atomIdx[atomCenterPair_targeted]

    atom_varNr_sel = atom_varNr[ atomCenterPair_atomIdx_targeted ]
    atom_atomNameNum_sel = atom_atomNameNum[ atomCenterPair_atomIdx_targeted ]
    atom_resnamenum_sel = atom_resnamenum[ atomCenterPair_atomIdx_targeted ]
    atomCenterPair_centerCoordsFlat_sel = atomCenterPair_centerCoordsFlat[ atomCenterPair_targeted ]
    atomCenterPair_vics_sel = atomCenterPair_vics[ atomCenterPair_targeted ]


    return atomCenterPair_atomIdx_targeted, \
           atom_varNr_sel,\
           atom_atomNameNum_sel,\
           atom_resnamenum_sel,\
           atomCenterPair_centerCoordsFlat_sel,\
           atomCenterPair_vics_sel



def generateAtomCenterPairs(atom_box_atomCenterCoords,
                            grid,
                            atom_box_coords,
                            atom_box_isTargetResidue,
                            atom_box_idx):

    ##############################
    # Generate atom-center pairs
    ##############################

    #printMemUsage("3-1-1")

    # <#atoms in box, #neighbors, 3>
    atomCenterPair_centerCoords = Voxelgrid.getCenterNeighbors(atom_box_atomCenterCoords, grid.neighborCreationSchedule)

    #printMemUsage("3-1-2")

    atomCenterPair_centerCoordsFlat = atomCenterPair_centerCoords.view(-1, 3)

    #printMemUsage("3-1-3")

    atomCenterPair_atomCoords = rep_3d(atom_box_coords, grid.neighborCreationSchedule.shape[0])  # .reshape(-1, 3)

    #printMemUsage("3-1-4")
    #atomCenterPair_isTargetResidue = rep_1d(atom_box_isTargetResidue, grid.neighborCreationSchedule.shape[0])  # .reshape(-1, 3)
    atomCenterPair_atomIdx = rep_1d(atom_box_idx, grid.neighborCreationSchedule.shape[0])

    #printMemUsage("3-1-5")

    atomCenterPair_goodCenterIdx_bool = torch.all((atomCenterPair_centerCoordsFlat <= grid.maxCenterCoord) & (atomCenterPair_centerCoordsFlat >= -grid.maxCenterCoord), dim=1)

    #printMemUsage("3-1-6")

    atomCenterPair_dists = ((atomCenterPair_centerCoordsFlat - atomCenterPair_atomCoords) ** 2).sum(axis=1)

    #printMemUsage("3-1-7")

    atomCenterPair_vics = 1 - (torch.sqrt(atomCenterPair_dists) / (grid.scanRadius))

    #printMemUsage("3-1-8")

    atomCenterPair_goodDistIdx_bool = atomCenterPair_dists < grid.scanRadius ** 2

    #printMemUsage("3-1-9")

    atomCenterPair_goodIdx_bool = atomCenterPair_goodCenterIdx_bool & atomCenterPair_goodDistIdx_bool
    # atomCenterPair_goodIdxJigsawEnv_bool = atomCenterPair_goodIdx_bool & ~atomCenterPair_isTargetResidue
    # atomCenterPair_goodIdxJigsawRes_bool = atomCenterPair_goodIdx_bool & atomCenterPair_isTargetResidue

    #printMemUsage("3-1-10")

    return  atomCenterPair_centerCoordsFlat, \
            atomCenterPair_vics, \
            atomCenterPair_goodIdx_bool, \
            atomCenterPair_atomIdx



def reduceToBox(atom_coords_centered,
                grid,
                atom_isTargetResidue,
                atom_idx):
    # # Find indices of atoms in box; return int index
    atom_boxAtomIdxs_bool = torch.all((atom_coords_centered < grid.edgeLen) & (atom_coords_centered > -grid.edgeLen), dim=1)
    atom_boxAtomIdxs = torch.where(atom_boxAtomIdxs_bool)[0]

    #Create box arrays
    atom_box_coords = atom_coords_centered[atom_boxAtomIdxs]
    atom_box_isTargetResidue = atom_isTargetResidue[atom_boxAtomIdxs]

    atom_box_idx = atom_idx[ atom_boxAtomIdxs ]

    return atom_box_coords, \
           atom_box_isTargetResidue, \
           atom_box_idx



def reduceToGood(goodIdx_bool,
                 atomCenterPair_centerCoordsFlat,
                 atomCenterPair_vics,
                 atomCenterPair_atomIdx,
                 grid):
    goodIdx = torch.where(goodIdx_bool)[0]

    atomCenterPair_sel_centerCoordsFlat = atomCenterPair_centerCoordsFlat[goodIdx]
    atomCenterPair_sel_vics = atomCenterPair_vics[goodIdx]
    atomCenterPair_sel_atomBoxIdx = atomCenterPair_atomIdx[goodIdx]

    return atomCenterPair_sel_centerCoordsFlat, \
           atomCenterPair_sel_vics, \
           atomCenterPair_sel_atomBoxIdx


def allocateOutput(grid,
                   c,
                   vars_changePoss):

    ##############################
    # Allocate output
    ##############################
    #nFeats = torch.tensor([1])

    outputDim_allAtom = [vars_changePoss.shape[0],
                 grid.nVoxels_oneDim,
                 grid.nVoxels_oneDim,
                 grid.nVoxels_oneDim,
                 1]

    outputDim_perAA = [vars_changePoss.shape[0],
                       grid.nVoxels_oneDim,
                       grid.nVoxels_oneDim,
                       grid.nVoxels_oneDim,
                       c["voxel_nAAs"] * c["nAtoms"]]

    voxels_proteinBatch_allAtom_jigsaw = torch.zeros(outputDim_allAtom, dtype=torch.float32, device=c["torch_device"])
    voxels_proteinBatch_perAA_jigsaw = torch.zeros(outputDim_perAA, dtype=torch.float32, device=c["torch_device"])

    voxels_proteinBatch_allAtom_full = torch.zeros(outputDim_allAtom, dtype=torch.float32, device=c["torch_device"])
    voxels_proteinBatch_perAA_full = torch.zeros(outputDim_perAA, dtype=torch.float32, device=c["torch_device"])

    voxels_proteinBatch_resId_jigsaw = torch.zeros(outputDim_allAtom, dtype=torch.int64, device=c["torch_device"])
    voxels_proteinBatch_resId_full = torch.zeros(outputDim_allAtom, dtype=torch.int64, device=c["torch_device"])

    return voxels_proteinBatch_allAtom_jigsaw, \
           voxels_proteinBatch_perAA_jigsaw, \
           voxels_proteinBatch_allAtom_full, \
           voxels_proteinBatch_perAA_full, \
           voxels_proteinBatch_resId_jigsaw, \
           voxels_proteinBatch_resId_full


def voxelize_scatter_allAtom(grid,
                             voxels_proteinBatch,
                             atomCenterPair_sel_centerCoordsFlat,
                             atomCenterPair_sel_varNr,
                             atomCenterPair_sel_vics):
    voxels_proteinBatch_flat = voxels_proteinBatch.ravel()

    ##############################
    # Scatter max
    ##############################

    # for t in [atomCenterPair_sel_centerCoordsFlat, atomCenterPair_sel_varIdxVec, gridSize]:
    #     print(t.device)

    gridSize = grid.nVoxels_oneDim * grid.nVoxels_oneDim * grid.nVoxels_oneDim
    atomCenterPair_sel_centerIdxs = grid.getCenterIdxs_batch(atomCenterPair_sel_centerCoordsFlat, atomCenterPair_sel_varNr, gridSize)

    _, voxels_proteinBatch_flat_argmax = scatter_max(atomCenterPair_sel_vics, atomCenterPair_sel_centerIdxs, out=voxels_proteinBatch_flat)

    return voxels_proteinBatch_flat_argmax


def voxelize_scatter_perAA(  grid,
                             c,
                             voxels_proteinBatch_perAA,
                             atomCenterPair_sel_atomIdxs,
                             atomCenterPair_sel_centerCoordsFlat,
                             atomCenterPair_sel_varNr,
                             atomCenterPair_sel_vics,
                             atomCenterPair_sel_atomNameNum,
                             atomCenterPair_sel_aaNameNum):

    #printMemUsage("6-3-1")

    #Take the box atom Idxs vector from the atom-center pairs; use to get corresponding atom types;
    #atomCenterPair_sel_atomNamesNum = atom_box_atomNamesNum[ atomCenterPair_sel_atomBoxIdxs ]

    #Get boolean index into pairs indicating whether its a target atom
    isTargetAtom_idx_bool = atomCenterPair_sel_atomNameNum > -1

    #printMemUsage("6-3-2")

    isTargetAtom_idx = torch.where(isTargetAtom_idx_bool)[0]

    #printMemUsage("6-3-3")

    #Reduce vectors to target elements
    atomCenterPair_sel_atomIdxs_targeted =  atomCenterPair_sel_atomIdxs[ isTargetAtom_idx ]
    atomCenterPair_sel_centerCoordsFlat_targeted = atomCenterPair_sel_centerCoordsFlat[ isTargetAtom_idx ]
    atomCenterPair_sel_atomNamesNum_targeted = atomCenterPair_sel_atomNameNum[ isTargetAtom_idx ]
    atomCenterPair_sel_aaNamesNum_targeted = atomCenterPair_sel_aaNameNum[ isTargetAtom_idx ]
    atomCenterPair_sel_vics_targeted = atomCenterPair_sel_vics[ isTargetAtom_idx ]
    atomCenterPair_sel_varIdxVec_targeted = atomCenterPair_sel_varNr[ isTargetAtom_idx ]

    #printMemUsage("6-3-4")

    voxels_proteinBatch_perAA_flat = voxels_proteinBatch_perAA.ravel()

    #printMemUsage("6-3-5")

    gridSize = grid.nVoxels_oneDim * grid.nVoxels_oneDim * grid.nVoxels_oneDim * c["voxel_nAAs"] * c["nAtoms"]

    #printMemUsage("6-3-6")

    atomCenterPair_sel_centerIdxsPerAA = grid.getCenterIdxs_perAtomAa_batch(   atomCenterPair_sel_centerCoordsFlat_targeted,
                                                                             atomCenterPair_sel_aaNamesNum_targeted,
                                                                             atomCenterPair_sel_atomNamesNum_targeted,
                                                                             atomCenterPair_sel_varIdxVec_targeted,
                                                                             gridSize)

    # import pandas as pd
    # df = pd.DataFrame({"atomCenterPair_sel_centerCoordsFlat_targeted": atomCenterPair_sel_centerCoordsFlat_targeted.tolist(),
    #                    "atomCenterPair_sel_aaNamesNum_targeted": atomCenterPair_sel_aaNamesNum_targeted.tolist(),
    #                    "atomCenterPair_sel_atomNamesNum_targeted": atomCenterPair_sel_atomNamesNum_targeted.tolist(),
    #                    "atomCenterPair_sel_varIdxVec_targeted": atomCenterPair_sel_varIdxVec_targeted.tolist(),
    #                    "atomCenterPair_sel_centerIdxsPerAA": atomCenterPair_sel_centerIdxsPerAA})
    #
    # print(df.sort_values("atomCenterPair_sel_centerIdxsPerAA"), flush=True)

    #printMemUsage("6-3-7")


    _, outputGrid_flat_argmax_perAA = scatter_max(atomCenterPair_sel_vics_targeted, atomCenterPair_sel_centerIdxsPerAA, out=voxels_proteinBatch_perAA_flat)

    #printMemUsage("6-3-8")

    return outputGrid_flat_argmax_perAA, \
           atomCenterPair_sel_atomIdxs_targeted



def getVoxelResIds_perAA(voxels_argmax_perAA,
                   atomCenterPair_sel_atomIdxs_targeted,
                   atom_residueIds,
                   atom_changePos):

    #printMemUsage("6-6-1")

    ##############################
    # Get winning idxs and build NN resid output
    ##############################

    # We determine which voxels (among _all_ voxels) is non-NA as a boolean index

    # print("voxels_argmax_perAA", voxels_argmax_perAA, flush=True)
    # print("voxels_argmax_perAA", voxels_argmax_perAA.shape, flush=True)

    # print("atomCenterPair_sel_atomIdxs_targeted", atomCenterPair_sel_atomIdxs_targeted, flush=True)
    # print("atomCenterPair_sel_atomIdxs_targeted", atomCenterPair_sel_atomIdxs_targeted.shape, flush=True)
    # print("atomCenterPair_sel_atomIdxs_targeted", atomCenterPair_sel_atomIdxs_targeted.shape[0], flush=True)

    voxels_argmax_centerIdx_perAA_bool = voxels_argmax_perAA < atomCenterPair_sel_atomIdxs_targeted.shape[0]

    # print("voxels_argmax_centerIdx_perAA_bool", voxels_argmax_centerIdx_perAA_bool, flush=True)
    # print("voxels_argmax_centerIdx_perAA_bool", voxels_argmax_centerIdx_perAA_bool.shape, flush=True)

    # printMemUsage("6-6-2")

    # Convert the boolean index to an integer index (--> center idxs)
    voxels_argmax_centerIdx_perAA = torch.where( voxels_argmax_centerIdx_perAA_bool )[0]

    #printMemUsage("6-6-3")

    # Retrieve the actual values of the non-NA entries, i.e. the indices pointing to rows in the list of all atom-center pairs
    voxels_argmax_selIdxs_perAA = voxels_argmax_perAA[voxels_argmax_centerIdx_perAA]

    #printMemUsage("6-6-4")

    # Get the atom indices of the non-NA entries using the index above
    voxels_argmax_atomIdxs_perAA = atomCenterPair_sel_atomIdxs_targeted[ voxels_argmax_selIdxs_perAA ]

    #printMemUsage("6-6-5")

    # Get the residue IDs of the non-NA entries using the index above
    voxels_argmax_residueIds_perAA = atom_residueIds[ voxels_argmax_atomIdxs_perAA ]
    #printMemUsage("6-6-6")

    voxels_argmax_changePoss_perAA = atom_changePos[ voxels_argmax_atomIdxs_perAA ]

    #printMemUsage("6-6-7")

    #Get the idx of the resIDs that are before in sequence
    voxels_argmax_beforeInSequence_perAA = torch.where(voxels_argmax_residueIds_perAA < voxels_argmax_changePoss_perAA)[0]
    #voxels_argmax_isChangePos = torch.where(voxels_argmax_residueIds == voxels_argmax_changePoss)[0]

    #printMemUsage("6-6-8")

    #Get the indices of the output tensor that need to be updated
    voxels_argmax_centerIdx_beforeInSequence_perAA = voxels_argmax_centerIdx_perAA[ voxels_argmax_beforeInSequence_perAA ]

    #printMemUsage("6-6-9")

    #Only relevant for the update from jigsaw to all atom
    #voxels_argmax_nonNaIdx_isChangePos_perAA = voxels_argmax_nonNaIdx_perAA

    return voxels_argmax_centerIdx_beforeInSequence_perAA, \
           voxels_argmax_centerIdx_perAA,\
           voxels_argmax_residueIds_perAA, \
           voxels_argmax_changePoss_perAA


def getVoxelResIds(voxels_argmax,
                   atomCenterPair_sel_atomIdxs,
                   atom_residueIds,
                   atom_batchResIds0Based,
                   atom_changePos):


    # We determine which voxels (among _all_ voxels) is non-NA as a boolean index
    voxels_argmax_nonNaIdx_bool = voxels_argmax < atomCenterPair_sel_atomIdxs.shape[0]

    # Convert the boolean index to an integer index (--> center idxs)
    voxels_argmax_centerIdx = torch.where(voxels_argmax_nonNaIdx_bool)[0]

    # Retrieve the actual values of the non-NA entries, i.e. the indices pointing to rows in the list of all atom-center pairs
    voxels_argmax_selIdxs = voxels_argmax[voxels_argmax_centerIdx]

    # Get the atom indices of the non-NA entries using the index above
    voxels_argmax_atomIdxs = atomCenterPair_sel_atomIdxs[ voxels_argmax_selIdxs ]

    # Get the residue IDs of the non-NA entries using the index above
    voxels_argmax_residueIds = atom_residueIds[ voxels_argmax_atomIdxs ]
    voxels_argmax_batchResIds0Based = atom_batchResIds0Based[voxels_argmax_atomIdxs]
    voxels_argmax_changePoss = atom_changePos[ voxels_argmax_atomIdxs ]


    #Get the idx of the resIDs that are before in sequence
    voxels_argmax_beforeInSequence = torch.where(voxels_argmax_residueIds < voxels_argmax_changePoss)[0]

    #Get the indices of the output tensor that need to be updated
    voxels_argmax_centerIdx_beforeInSequence = voxels_argmax_centerIdx[ voxels_argmax_beforeInSequence ]

    return voxels_argmax_centerIdx_beforeInSequence, \
           voxels_argmax_centerIdx, \
           voxels_argmax_residueIds, \
           voxels_argmax_batchResIds0Based


def rotate(atom_coords,
           atom_centralCalphaCoords,
           grid,
           vars_changePoss,
           nAtomsPerVar):

    atom_coords_centered = atom_coords - atom_centralCalphaCoords

    rotMatIndexVec = torch.randint(0, grid.rotMatrices.shape[0], (vars_changePoss.shape[0], 1))

    rotMats = grid.rotMatrices[rotMatIndexVec, :]

    rotMatsRep = rotMats.repeat_interleave(nAtomsPerVar, dim=0)

    atom_coords_centered = torch.bmm(atom_coords_centered.view(-1, 1, 3), rotMatsRep.view(-1, 3, 3))
    atom_coords_centered = atom_coords_centered.squeeze()

    return atom_coords_centered



def printMemUsage(msg):
    print(msg, torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024 ** 3, flush=True)





