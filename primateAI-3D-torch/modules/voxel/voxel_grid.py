import torch
from scipy.stats import special_ortho_group
import numpy as np
import pandas as pd

from helper.helper_voxel import rep_1d, rep_3d

class Voxelgrid:
    def __init__(self, c):
        self.c = c

        self.nVoxels = c["voxel_nVoxels"]                                      #torch.from_numpy(np.array(c["voxel_nVoxels"])).to(c["torch_device"])
        self.voxelSize = torch.from_numpy(np.array(c["voxel_voxelSize"])).to(c["torch_device"])
        self.scanRadius = torch.from_numpy(np.array(c["voxel_scanRadius"])).to(c["torch_device"])

        self.boxSize = [nVoxelsOneDim*self.voxelSize for nVoxelsOneDim in self.nVoxels]# * self.voxelSize
        self.edgeLen = (self.boxSize[0] / 2) + 4.6

        self.boxLen = self.nVoxels[0] * self.voxelSize
        self.nVoxels_oneDim = self.nVoxels[0]

        self.boxLen_half = self.boxLen / 2

        self.maxVoxelIdx = self.nVoxels_oneDim - 1

        self.rotMatrices = torch.from_numpy(self.initRotMatrices(c["voxel_rotate"])).to(c["torch_device"])

        self.centers_cpu = self.getGridCenters( c["voxel_nVoxels"], c["voxel_voxelSize"] )
        self.centers = torch.from_numpy(self.centers_cpu).to( c["torch_device"] )

        self.maxCenterCoord = torch.max(self.centers)

        self.neighborCreationSchedule = torch.from_numpy(self.getNeighborCreationSchedule(self.centers_cpu, self.c["voxel_scanRadius"])).to(c["torch_device"])



    def initRotMatrices(self, rotate):
        rotMatrices = []
        for i in range(10000):

            if rotate:
                rotMat = np.float32(special_ortho_group.rvs(3))
            else:
                rotMat = np.array( [[1.,  0.,  0.], \
                                    [0.,  1.,  0.], \
                                    [0.,  0.,  1.]], dtype=np.float32)

            rotMatrices.append(rotMat)

        rotMatrices_np = np.array(rotMatrices, dtype=np.float32)

        return rotMatrices_np


    def getGridCenters(self, nVoxels, voxelSize):

        center = np.array([0, 0, 0])

        if nVoxels[0] % 2 == 0:
            raise Exception("Number of voxels must be odd!")

        x, y, z = nVoxels

        firstdim = np.repeat(np.arange(x) * voxelSize, y* z)

        seconddim = np.tile(np.repeat(np.arange(y) * voxelSize, z), x)

        thirddim = np.tile(np.arange(z) * voxelSize, x * y)

        combined = np.vstack((firstdim.T, seconddim.T, thirddim.T)).T.astype(np.float64)
        combined = combined.reshape([x, y, z, 3])

        nVoxelsSide = (nVoxels[0] - 1) / 2
        minCenter = center - (nVoxelsSide * voxelSize)

        centers = combined + minCenter

        centers = centers.reshape(np.prod(nVoxels), 3).copy().astype(np.float32)

        return centers


    def getNeighborCreationSchedule(self, centers, scanRadius):
        distFromCenter = np.linalg.norm(centers, axis=1)
        neighborCreationSchedule = centers[ distFromCenter < scanRadius ]
        return neighborCreationSchedule


    def getCenterIdxs(self, centerCoords):
        centerIdxs_3D = (centerCoords / self.voxelSize) + ((self.nVoxels_oneDim - 1) / 2)
        centerIdxs_1D =  centerIdxs_3D[:,2] + \
                        (centerIdxs_3D[:,1] * self.nVoxels_oneDim) + \
                        (centerIdxs_3D[:,0] * self.nVoxels_oneDim * self.nVoxels_oneDim)
        centerIdxs_1D = centerIdxs_1D.to(torch.int64)
        return centerIdxs_1D


    def getCenterIdxs_perAtomAa(self, centerCoords, aaNamesNum, atomNamesNum):

        centerIdxs_1D = atomNamesNum + \
            self.c["nAtoms"] * aaNamesNum + \
            self.c["nAtoms"] * self.c["voxel_nAAs"] * self.getCenterIdxs(centerCoords)

        return centerIdxs_1D





    def getCenterIdxs_batch(self, centerCoords, varIdx, gridSize):

        centerIdxs_1D = varIdx * gridSize + \
                        self.getCenterIdxs(centerCoords)

        return centerIdxs_1D


    def getCenterIdxs_perAtomAa_batch(self, centerCoords, aaNamesNum, atomNamesNum, varIdx, gridSize):

        centerIdxs_1D = varIdx * gridSize + \
            self.getCenterIdxs_perAtomAa(centerCoords, aaNamesNum, atomNamesNum)

        return centerIdxs_1D


    @staticmethod
    def getCenterNeighbors(centerCoords, neighborCreationSchedule):
        atomCenterNeighbors = rep_3d(centerCoords, neighborCreationSchedule.shape[0]).view(centerCoords.shape[0], -1, 3)
        atomCenterNeighbors = atomCenterNeighbors + neighborCreationSchedule
        return atomCenterNeighbors


    @staticmethod
    def getNearestBoxCenter(atomCoords, voxelSize, maxCenterCoord):
        atomCenterCoords = torch.round((atomCoords / voxelSize)) * voxelSize
        nearestCenterCoords = torch.maximum(-maxCenterCoord, torch.minimum(maxCenterCoord, atomCenterCoords))
        return nearestCenterCoords


    def toDF(self):

        centers = self.centers.cpu()

        df = pd.DataFrame({"center_x": centers[:,0],
                           "center_y": centers[:,1],
                           "center_z": centers[:,2]
                           })

        return df