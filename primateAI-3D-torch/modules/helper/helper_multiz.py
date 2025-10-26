import pandas as pd
import numpy as np

class Multiz:
    def __init__(self, c):
        self.c = c
        self.generateSpeciesFilter()

    def generateSpeciesFilter(self):
        self.speciesDF = pd.read_csv(self.c["multiz_speciesDfFilePath"])

        if not "multizGapStopFeats" in self.c:
            self.c["multiz_gapStopFeats"] = False
        self.multizGapStopFeats = self.c["multiz_gapStopFeats"]

        boolMasksAliType = []
        if self.c["multiz_isZoonomia"]:
            boolMasksAliType.append(self.speciesDF["isZoonomia"])
        if self.c["multiz_isZ100"]:
            boolMasksAliType.append(self.speciesDF["isZ100"])
        if self.c["multiz_isJack"]:
            boolMasksAliType.append(self.speciesDF["isJack"])
        boolMaskAliType = pd.DataFrame(boolMasksAliType).any().values

        boolMasksPrimate = []
        if self.c["multiz_isPrimate"]:
            boolMasksPrimate.append(self.speciesDF["isPrimate"])
        if self.c["multiz_isNonPrimate"]:
            boolMasksPrimate.append(self.speciesDF["isNonPrimate"])
        boolMaskPrimate = pd.DataFrame(boolMasksPrimate).any().values

        self.boolMaskSpecies = boolMaskAliType & boolMaskPrimate

        if not self.c["multiz_isHuman"]:
            boolMaskHuman = (self.speciesDF.tax_id == 9606).copy().values
            self.boolMaskSpecies = self.boolMaskSpecies & (~boolMaskHuman)

        self.nSpecies = self.boolMaskSpecies.astype(int).sum()

        assert (self.nSpecies) > 0

    def filterAndCombineMultiz(self, pdbDict):
        """ Add to pdbDict the tensor multiz_seqArray. For now, other tensors
        are ignored/not supported.
        """
        for i, (geneName, geneDict) in enumerate(pdbDict.items()):
            if i % 1000 == 0: print(i)
            #print(geneName)
            geneDict["multiz_seqArray"] = geneDict["multiz_seqArray"][:, self.boolMaskSpecies]

    def computeNumberAAs(self, pdbDict):
        # multiz_seqArray contains AA coded as integers from 0 to 19 as well as
        # additional special integers for gaps, etc.
        # we assume that all these integers go from 0 to n_aas, and we take the
        # max to be n_aas.

        max_aa = 0
        for geneDict in pdbDict.values():
            max_aa = max(max_aa,
                         geneDict["multiz_seqArray"][ (geneDict["multiz_seqArray"] >= 0) & (geneDict["multiz_seqArray"] <= 30) ].max() )


        for geneDict in pdbDict.values():
            geneDict["multiz_seqArray"] = np.where(
                geneDict["multiz_seqArray"] > max_aa,
                max_aa + 1,
                geneDict["multiz_seqArray"])

        self.n_aas = max_aa + 2
        print("Number of aas in multiz_seqArray", self.n_aas)

