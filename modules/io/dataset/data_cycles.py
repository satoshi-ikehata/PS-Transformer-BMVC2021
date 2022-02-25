import glob
import os,sys
import cv2
import numpy as np
import math
# from . import openexr_io
from numpy import unique
# from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt

class io():
    def __init__(self, data_root):
        self.data_root = data_root
        self.img_channels = 3
        self.datatype = 'Cycles'
        self.dirname = ['CyclesPS_Dichromatic', 'CyclesPS_Metallic', 'CyclesPS_Dichromatic', 'CyclesPS_Metallic']
        self.suffix= ['direct.tif', 'direct.tif','indirect.tif', 'indirect.tif']
        if len(self.dirname) != len(self.suffix):
            raise Exception("dirname and suffix have different length")
        self.ext = '.obj'
        self.objlists = []
        for i in range(len(self.dirname)):
            data_path = f'{data_root}/{self.dirname[i]}'
            objlist = []
            [objlist.append(p) for p in glob.glob(data_path + '/*%s' % self.ext, recursive=True) if os.path.isdir(p)]
            objlist = sorted(objlist)
            self.objlists.append(objlist)

    def get_num_object(self):
        return len(self.objlists[0])

    def get_num_set(self):
        return len(self.objlists)

    def load(self, objid, objset, sizeImgBuffer=None, scale=1.0):
        objlist = self.objlists[objset]
        objname = objlist[objid].split('/')[-1]
        imglist = []
        [imglist.append(p) for p in glob.glob(objlist[objid] + '/*_%s' % self.suffix[objset], recursive=True) if os.path.isfile(p)]
        imglist = sorted(imglist)
        if len(imglist) == 0:
            return False

        if os.name == 'posix':
            temp = imglist[0].split("/")
        if os.name == 'nt':
            temp = imglist[0].split("\\")
        img_dir = "/".join(temp[:-1])
        print(f'Loading {objname} / {self.dirname[objset]}, {self.suffix[objset]} (Cycles)')

        if sizeImgBuffer is not None:
            indexset = np.random.permutation(len(imglist))[:sizeImgBuffer]
        else:
            indexset = range(len(imglist))
        I = []
        for i, indexofimage in enumerate(indexset):
            img_path = imglist[indexofimage]
            img = cv2.resize(cv2.cvtColor(cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB), dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_NEAREST)
            if img.dtype == 'uint8':
                bit_depth = 255.0
            if img.dtype == 'uint16':
                bit_depth = 65535.0

            img = np.float32(img) / bit_depth
            h = img.shape[0]
            w=h
            I.append(img)
            nml_path = img_dir + '/gt_normal.tif'
            if os.path.isfile(nml_path) and i == 0:
                N = np.float32(cv2.resize(cv2.cvtColor(cv2.imread(nml_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB), dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_NEAREST))/65535.0
                N = 2 * N - 1
                mask = np.abs(1 - np.sqrt(np.sum(N*N, axis=2))) < 1.0e-2
                N /= np.sqrt(np.sum(N*N, axis=2).reshape(N.shape[0], N.shape[1], 1))
                N = N * mask.reshape(N.shape[0], N.shape[1], 1)
                N = np.reshape(N, (h * w, 3))

        I = np.array(I)
        L = np.loadtxt(img_dir + '/light.txt', np.float32) # N x 3
        L = L[indexset,:]
        M = mask.reshape(-1,1)
        ids = np.nonzero(mask)
        valid_u = ids[1]
        valid_v = ids[0]
        valid = ids[0] * w + ids[1]
        I = np.reshape(I, (-1, h * w, 3))
        I = np.transpose(I, (1, 2, 0))
        return I, N, L, M, valid, h, w
