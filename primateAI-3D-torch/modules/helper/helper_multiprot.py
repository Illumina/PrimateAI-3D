
class MultiProt():
    def __init__(self, c, protList):
        self.c = c
        self.protList = protList

    def toDevice_labels(self):
        for subprot in self.protList:
            subprot.toDevice_labels()

def addScoresToSubprots(protList, scoreDict, scope=["protein", "dna"]):
    # print("Updating")

    currIndex = 0
    for prot in protList:
        for scoreName, scoreTensor in scoreDict.items():
            # print(scoreDict[scoreName][currIndex:currIndex + prot.changePoss.shape[0], :].shape)

            prot_batch_scores = scoreDict[scoreName][currIndex:currIndex + prot.changePoss.shape[0], :]

            # print(prot.scoreDict[scoreName][prot.dna_batchPos0based, prot.dna_batch_altAaNum].unsqueeze(1).shape)

            if "protein" in scope:

                prot.outputDict["protein"]["score"][scoreName] = prot_batch_scores

            if "dna" in scope:
                dna_batchPos0based = prot.outputDict["dna"]["label"]["batchPos0based"]
                dna_altAaNum = prot.outputDict["dna"]["label"]["aaNumAlt"]

                dna_batch_scores = prot_batch_scores[dna_batchPos0based, dna_altAaNum].unsqueeze(1)

                prot.outputDict["dna"]["score"][scoreName] = dna_batch_scores

        currIndex += prot.changePoss.shape[0]

