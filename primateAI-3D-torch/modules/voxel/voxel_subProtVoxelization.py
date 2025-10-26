def getVoxelDict(subprot, device):
    voxelDict = {"atom_coords": subprot.targetPdbObject.get('atom_coords').to(device),
                 "atom_atomNamesNum": subprot.targetPdbObject.get('atom_atomNamesNum').to(device),
                 "atom_resid": subprot.targetPdbObject.get('atom_resid').to(device),
                 "atom_resnamenum": subprot.targetPdbObject.get('atom_resnamenum').to(device),
                 "caCoords": subprot.targetPdbObject.get('caCoords').to(device),
                 "caAllAtomIndex": subprot.targetPdbObject.get('caAllAtomIndex').to(device)}

    return voxelDict


def getCopyTensor(voxelDict, nCopies, resOffset):
    batch_atom_coords = voxelDict["atom_coords"].repeat(nCopies, 1, 1) #coordinates of all atoms
    batch_atom_resid = voxelDict["atom_resid"].repeat(nCopies) #change_position_1based for each atom
    batch_atom_atomNamesNum = voxelDict["atom_atomNamesNum"].repeat(nCopies) #atom types in numeric representation, in particular to distinguish atoms that we are about (e.g. Calpha) from those that we do not
    batch_atom_resnamenum = voxelDict["atom_resnamenum"].repeat(nCopies) #amino acid type indicator for each atom

    batch_atom_batchResIds0Based = (batch_atom_resid - 1) + resOffset

    batch_atom_coords_flat = batch_atom_coords.view(-1, 3)

    return batch_atom_coords_flat, \
           batch_atom_resid, \
           batch_atom_atomNamesNum, \
           batch_atom_resnamenum, \
           batch_atom_batchResIds0Based

