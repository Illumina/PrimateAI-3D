import pickle
import lz4.frame
import globalVars
import numpy as np
import keras
from keras import backend as K
import traceback

def getLmdbRecs(geneId, change_positions_1based):
    idToGeneName = globalVars.globalVars["idToGeneName"]

    gene_name = idToGeneName[geneId]

    multizLmdb = globalVars.globalVars["multizLmdb"]

    multizLmdbWrapper = globalVars.globalVars["multizLmdbWrapper"]
    nSpecies = multizLmdbWrapper.nSpecies
    nSpeciesTotal = multizLmdbWrapper.nSpeciesTotal
    speciesMask = multizLmdbWrapper.boolMask
    multizGapStopFeats = multizLmdbWrapper.multizGapStopFeats

    retArr = np.full((change_positions_1based.shape[0], 3, nSpecies), 21, dtype=np.uint8)

    with multizLmdb.begin() as lmdbTxn:

        for i in range(change_positions_1based.shape[0]):
            change_position_1based = change_positions_1based[i]

            targetKey = "%s_%s" % (gene_name, change_position_1based)

            targetTupleObjectBytes = lmdbTxn.get(targetKey.encode("ascii"))

            if targetTupleObjectBytes != None:
                targetTupleObject = pickle.loads(lz4.frame.decompress(targetTupleObjectBytes))
                seqs, gapFracs, containsStop = targetTupleObject
                seqs = np.where(seqs > 20, 20, seqs)

                if not multizGapStopFeats:
                    gapFracs = np.full((nSpeciesTotal), 0, dtype=np.uint8)
                    containsStop = np.full((nSpeciesTotal), 0, dtype=np.uint8)

            else:
                seqs = np.full((nSpeciesTotal), 22, dtype=np.uint8)
                gapFracs = np.full((nSpeciesTotal), 0, dtype=np.uint8)
                containsStop = np.full((nSpeciesTotal), 0, dtype=np.uint8)

            try:
                retArr[i, 0, :] = seqs[speciesMask]
                retArr[i, 1, :] = gapFracs[speciesMask]
                retArr[i, 2, :] = containsStop[speciesMask]
            except:
                print(gene_name, change_position_1based)
                traceback.print_exc()
                raise Exception("Error")

    return retArr


def getLmdbGeneRecs(geneId):
    idToGeneName = globalVars.globalVars["idToGeneName"]

    gene_name = idToGeneName[geneId]

    multizLmdb = globalVars.globalVars["multizLmdb"]

    multizLmdbWrapper = globalVars.globalVars["multizLmdbWrapper"]
    nSpecies = multizLmdbWrapper.nSpecies
    speciesMask = multizLmdbWrapper.boolMask
    multizGapStopFeats = multizLmdbWrapper.multizGapStopFeats

    retArrGlobal = np.full((3, nSpecies), 0, dtype=np.uint8)

    if multizGapStopFeats:
        with multizLmdb.begin() as lmdbTxn:
            targetGlobalObjectBytes = lmdbTxn.get(gene_name.encode("ascii"))
            if targetGlobalObjectBytes != None:
                retArrGlobal = pickle.loads(lz4.frame.decompress(targetGlobalObjectBytes))
                retArrGlobal = np.transpose(retArrGlobal)
                retArrGlobal = retArrGlobal[:, speciesMask]

    return retArrGlobal[np.newaxis,:]

def getLocalResIdMapping(uniqueResIds):
    maxResId = K.tf.reduce_max(uniqueResIds)

    uniqueResIds_exp = K.expand_dims(uniqueResIds, -1)

    changePossToLocalPoss_updates = K.tf.range(1, K.shape(uniqueResIds)[0] + 1)

    returnVal = K.tf.scatter_nd(
        uniqueResIds_exp,
        changePossToLocalPoss_updates,
        [maxResId + 1])

    fillTensor = K.tf.fill(K.shape(returnVal), K.constant(40000.0))
    returnVal = K.cast(returnVal, dtype=K.tf.float32)
    returnVal = K.tf.where(K.equal(returnVal, 0), fillTensor, returnVal - 1)

    return returnVal


def voxelGrid_changePosToLocalPos(changePoss_unique, multiz_voxelGridNN):
    uniqueResIdsToIdx = getLocalResIdMapping(changePoss_unique)

    voxelGrid_changePoss = K.reshape(multiz_voxelGridNN, (-1, 1))[:, 0]

    voxelGrid_centerIdx = K.tf.range(K.shape(voxelGrid_changePoss)[0])

    voxelGrid_noNaMask = K.not_equal(voxelGrid_changePoss, -1)

    voxelGrid_changePoss_noNA = K.tf.boolean_mask(voxelGrid_changePoss, voxelGrid_noNaMask)
    voxelGrid_centerIdx_noNA = K.tf.boolean_mask(voxelGrid_centerIdx, voxelGrid_noNaMask)
    voxelGrid_centerIdx_noNA_exp = K.expand_dims(voxelGrid_centerIdx_noNA, -1)

    voxelGrid_changePoss_noNA_local = K.tf.gather(uniqueResIdsToIdx, voxelGrid_changePoss_noNA)
    voxelGrid_changePoss_noNA_local = voxelGrid_changePoss_noNA_local + 1
    voxelGrid_changePoss_noNA_local = K.cast(voxelGrid_changePoss_noNA_local, dtype=K.tf.int32)

    voxelGrid_changePoss_local = K.tf.scatter_nd(voxelGrid_centerIdx_noNA_exp,
                                               voxelGrid_changePoss_noNA_local,
                                               K.shape(voxelGrid_changePoss)) - 1

    return voxelGrid_changePoss_local


def getChangePossUnique(multiz_voxelGridNN):
    changePoss = K.reshape(multiz_voxelGridNN, (-1, 1))[:, 0]
    changePoss_unique, idx = K.tf.unique(changePoss)
    changePoss_unique = K.tf.boolean_mask(changePoss_unique, K.not_equal(changePoss_unique, -1))

    return changePoss_unique


def repeat(t, n):
    t_new = K.tile(t, [n])
    t_new = K.reshape(t_new, (-1, K.shape(t)[0]))
    t_new = K.tf.transpose(t_new)
    t_new = K.reshape(t_new, [-1])

    return t_new


def seqArrToIdxArr(seqArr):
    idx_pos = repeat(K.tf.range(K.shape(seqArr)[0]), K.shape(seqArr)[1])
    idx_species = K.tile(K.tf.range(K.shape(seqArr)[1]), [K.shape(seqArr)[0]])

    idx = K.stack([idx_pos, idx_species, K.reshape(seqArr, [-1])])
    idx = K.tf.transpose(idx)

    idx = K.reshape(idx, [K.shape(seqArr)[0], K.shape(seqArr)[1], -1])

    return idx

def expandLmdbArr(lmdbArr, index, shape):
    lmdbArr_enc = K.tf.scatter_nd(index,
                               lmdbArr,
                               shape)

    lmdbArr_enc = K.tf.transpose(lmdbArr_enc, perm=[0, 2, 1])
    lmdbArr_enc = K.expand_dims(lmdbArr_enc, 0)
    lmdbArr_enc = K.cast(lmdbArr_enc, dtype=K.tf.float32)

    return lmdbArr_enc

def encodeLmdbArr(lmdbArr, lmdbArrGlobal, changePoss_unique):

    nPos = K.maximum(K.shape(changePoss_unique)[0], 1)

    lmdbArr = K.reshape(lmdbArr, (nPos, 3, -1))

    lmdbArrGlobal = K.tile(lmdbArrGlobal, [nPos, 1, 1])

    #SEQARR
    seqArr = lmdbArr[:, 0, :]

    seqArr_shape = [K.shape(seqArr)[0], K.shape(seqArr)[1], 23]

    seqArr_update = K.ones_like(seqArr)

    seqArr_index = seqArrToIdxArr(seqArr)

    seqArr_enc = K.tf.scatter_nd(seqArr_index,
                               seqArr_update,
                               seqArr_shape)

    seqArr_enc = K.tf.transpose(seqArr_enc, perm=[0, 2, 1])
    seqArr_enc = K.expand_dims(seqArr_enc, 0)
    seqArr_enc = K.cast(seqArr_enc, dtype=K.tf.float32)

    #GAPFRACS
    gapArr = lmdbArr[:, 1, :]
    gapArr_enc = expandLmdbArr(gapArr, seqArr_index, seqArr_shape)
    gapArr_enc = gapArr_enc / 254.0

    #HASSTOP
    stopArr = lmdbArr[:, 2, :]
    stopArr_enc = expandLmdbArr(stopArr, seqArr_index, seqArr_shape)

    # GLOBAL ALL GAP
    allgapArr = lmdbArrGlobal[:, 0, :]
    allgapArr_enc = expandLmdbArr(allgapArr, seqArr_index, seqArr_shape)

    # GLOBAL HAS STOP
    hasstopArr = lmdbArrGlobal[:, 1, :]
    hasstopArr_enc = expandLmdbArr(hasstopArr, seqArr_index, seqArr_shape)

    # GLOBAL GAP FRAC
    gapGlobalArr = lmdbArrGlobal[:, 2, :]
    gapGlobalArr_enc = expandLmdbArr(gapGlobalArr, seqArr_index, seqArr_shape)
    gapGlobalArr_enc = gapGlobalArr_enc / 254.0


    retArr_enc = K.concatenate([seqArr_enc, gapArr_enc, stopArr_enc, allgapArr_enc, hasstopArr_enc, gapGlobalArr_enc], axis=-1)





    return retArr_enc


def getMultizEnc(geneId, multiz_voxelGridNN):
    geneId = geneId[0]

    changePoss_unique = getChangePossUnique(multiz_voxelGridNN)

    multiz_voxelGridNN_local = voxelGrid_changePosToLocalPos(changePoss_unique, multiz_voxelGridNN)

    lmdbArr = K.tf.py_func(getLmdbRecs, [geneId, changePoss_unique], K.tf.uint8, stateful=False)
    lmdbArrGlobal = K.tf.py_func(getLmdbGeneRecs, [geneId], K.tf.uint8, stateful=False)

    lmdbArr = K.cast(lmdbArr, dtype='int32')
    lmdbArrGlobal = K.cast(lmdbArrGlobal, dtype='int32')
    
    seqArr_enc = encodeLmdbArr(lmdbArr, lmdbArrGlobal, changePoss_unique)

    return seqArr_enc, multiz_voxelGridNN_local


def mergeVoxelGridConv(multiz_voxelGridNN_local, seqArr_enc_conv):
    seqArr_enc_conv = K.reshape(seqArr_enc_conv, [K.shape(seqArr_enc_conv)[1], K.shape(seqArr_enc_conv)[2] * K.shape(seqArr_enc_conv)[3]] )

    voxelGrid_centerIdx = K.tf.range(K.shape(multiz_voxelGridNN_local)[0])

    voxelGrid_noNaMask = K.not_equal(multiz_voxelGridNN_local, -1)

    multiz_voxelGridNN_local_noNA = K.tf.boolean_mask(multiz_voxelGridNN_local, voxelGrid_noNaMask)

    voxelGrid_centerIdx_noNA = K.tf.boolean_mask(voxelGrid_centerIdx, voxelGrid_noNaMask)
    voxelGrid_centerIdx_noNA_exp = K.expand_dims(voxelGrid_centerIdx_noNA, -1)



    multiz_voxelGrid_conv_noNA = K.tf.gather(seqArr_enc_conv, multiz_voxelGridNN_local_noNA)

    outputShape = [K.shape(multiz_voxelGridNN_local)[0], K.shape(multiz_voxelGrid_conv_noNA)[1]]

    multiz_voxelGrid_conv = K.tf.scatter_nd(voxelGrid_centerIdx_noNA_exp,
                                          multiz_voxelGrid_conv_noNA,
                                          outputShape)



    return multiz_voxelGrid_conv

class MultizLayer(keras.layers.Layer):
    def __init__(self, nSpecies, nVoxels, nAAs, nConvFilters, convDoRelu, **kwargs):  # , units=32, input_dim=32
        super(MultizLayer, self).__init__(**kwargs)
        self.nSpecies = nSpecies
        self.nAAs = nAAs
        self.nVoxels = nVoxels
        self.nConvFilters = nConvFilters
        self.convDoRelu = convDoRelu
        self.trainable = True


    def build(self, input_shape_list):

        self.kernel = self.add_weight(name="kernel", shape = (1, 1, self.nSpecies*6, self.nConvFilters), trainable=True,  initializer='glorot_uniform')
        self.bias = self.add_weight(name="bias", shape = (self.nConvFilters,), trainable=True, initializer='glorot_uniform')

        #super(MultizLayer, self).build(input_shape_list)

    def doConv(self, inputs):
        outputs = K.conv2d(inputs, self.kernel)
        outputs = K.bias_add(outputs, self.bias)
        if self.convDoRelu:
            outputs = K.relu(outputs)

        return outputs


    def get_config(self):
        config = super(MultizLayer, self).get_config()
        return config

    def procVar(self, inputTuple):
        geneId, multiz_voxelGridNN, i = inputTuple

        bad_mult = K.constant([-40000], dtype=K.tf.int32)
        multiz_voxelGridNN = K.tf.where(K.equal(multiz_voxelGridNN, -1), multiz_voxelGridNN * bad_mult, multiz_voxelGridNN)

        seqArr_enc, multiz_voxelGridNN_local = getMultizEnc(geneId, multiz_voxelGridNN)

        seqArr_enc = K.reshape(seqArr_enc, [K.shape(seqArr_enc)[0], K.shape(seqArr_enc)[1], K.shape(seqArr_enc)[2], self.nSpecies*6])

        seqArr_enc_conv = self.doConv(seqArr_enc)

        multiz_voxelGrid_conv = mergeVoxelGridConv(multiz_voxelGridNN_local, seqArr_enc_conv)

        multiz_voxelGrid_conv = K.reshape(multiz_voxelGrid_conv, [self.nVoxels, self.nVoxels, self.nVoxels, self.nAAs*self.nConvFilters])

        return multiz_voxelGrid_conv

    def call(self, inputs):
        multiz_geneIds, multiz_voxelGridNNs_val = inputs
        iss = K.tf.range(K.shape(multiz_geneIds)[0])

        multiz_geneIds = K.cast(multiz_geneIds, dtype=K.tf.int32)
        multiz_voxelGridNNs_val = K.cast(multiz_voxelGridNNs_val, dtype=K.tf.int32)

        multiz_voxelGrid_conv = K.tf.map_fn(self.procVar, (multiz_geneIds, multiz_voxelGridNNs_val, iss), dtype=K.tf.float32)  #, parallel_iterations=1

        multiz_voxelGrid_conv = K.reshape(multiz_voxelGrid_conv, [-1, self.nVoxels, self.nVoxels, self.nVoxels, self.nAAs*self.nConvFilters])

        return multiz_voxelGrid_conv

    def compute_output_shape(self, input_shape_list):

        inputShape_geneIds, input_shape_voxelGrid = input_shape_list

        return (inputShape_geneIds[0], self.nVoxels, self.nVoxels, self.nVoxels, self.nAAs*self.nConvFilters)

    def get_config(self):
        base_config = super(MultizLayer, self).get_config()

        base_config['nSpecies'] = self.nSpecies
        base_config['nAAs'] = self.nAAs
        base_config['nVoxels'] = self.nVoxels
        base_config['nConvFilters'] = self.nConvFilters
        base_config['convDoRelu'] = self.convDoRelu
        base_config['trainable'] = self.trainable

        return base_config
