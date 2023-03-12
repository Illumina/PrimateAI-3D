from keras.callbacks import Callback
import numpy as np

class MeanPvalEval(Callback):
    def __init__(self, callbacks, verbose=0):
        super(MeanPvalEval, self).__init__()
  
        self.callbacks = callbacks


    def performEval(self, model, savePredFile=None):


        sigDictAll = {}
        sigScores = []
        sigScoresJigsaw = []
        sigScoresBoth = []
        for callbacki in self.callbacks:
            y_df, sigDict = callbacki.performEval(model)
            for keyi, vali in sigDict.items():
                if "sig_score_raw_alt" in keyi:
                    print("Averaging %s" % keyi)
                    sigScores.append( vali )
                if "sig_score_rawJigsaw_alt" in keyi:
                    print("Averaging %s" % keyi)
                    sigScoresJigsaw.append( vali )
                if "sig_score_rawBoth_alt" in keyi:
                    print("Averaging %s" % keyi)
                    sigScoresBoth.append( vali )
                sigDictAll[ keyi ] = vali

        sigDictAll["sig_score_raw_alt_mean"] = np.mean(np.array(sigScores))
        sigDictAll["sig_score_rawJigsaw_alt_mean"] = np.mean(np.array(sigScoresJigsaw))
        sigDictAll["sig_score_rawBoth_alt_mean"] = np.mean(np.array(sigScoresBoth))

        return None, sigDictAll

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        y_df, sigDict = self.performEval(self.model)

        for sigName, sigVal in sorted(sigDict.items(), key=lambda x: x[0]):
            print(sigName, sigVal)
            logs[sigName] = sigVal
            self.history.setdefault(sigName, []).append(sigVal)


        
        
        
