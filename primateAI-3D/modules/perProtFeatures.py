import numpy as np
from keras.utils import np_utils

def encodeRefAlt(snpRow):
    refAltEnc = np.concatenate([np_utils.to_categorical(snpRow.label_numeric_aa, num_classes=21), np_utils.to_categorical(snpRow.label_numeric_aa_alt, num_classes=21)])

    return refAltEnc[np.newaxis, :]
