import glob
import os,sys
import torch.utils.data as data
from .dataset import data_cycles
from .dataset import data_diligent
from .patch import *
from .observation_map import *

class custom_dataloader(data.Dataset):
    def __init__(self, data_type, data_root, num_samples=-1, min_nimg=10, max_nimg=10, block_size = 8):

        self.data_type = data_type
        self.min_nimg = min_nimg
        self.max_nimg = max_nimg
        self.num_samples = num_samples
        self.block_size = block_size

        if self.data_type == 'Cycles':
            self.data = data_cycles.io(data_root)
        elif self.data_type == 'DiLiGenT':
            self.data = data_diligent.io(data_root)
        else:
            raise Exception("Invalid Data Type")

        self.I = []
        self.N = []
        self.L = []
        self.M = []
        self.valid = []
        self.data_count = 0
        self.data_id = []
        self.h = []
        self.w = []

    def reset(self):
        self.I = []
        self.N = []
        self.L = []
        self.M = []
        self.valid = []
        self.data_count = 0
        self.data_id = []
        self.h = []
        self.w = []

    def load(self, objid, objset=0):
        self.data_count += 1

        if self.data_type == 'Cycles':
            I, N, L, M, valid, h, w = self.data.load(objid, objset)
            self.I.append(I)
            self.N.append(N)
            self.L.append(L)
            self.M.append(M)
            self.h.append(h)
            self.w.append(w)

            self.valid.append(valid)
            self.data_id.append((self.data_count-1) * np.ones((valid.shape[0],), np.int32))
            self.data_id_array = np.concatenate(self.data_id, axis=0)
            self.valid_array = np.concatenate(self.valid, axis=0)

            if self.num_samples > 0:
                sample_id = np.random.permutation(len(self.valid_array))[:self.num_samples]
                self.data_id_array = self.data_id_array[sample_id]
                self.valid_array = self.valid_array[sample_id]

        elif self.data_type == 'DiLiGenT':
            I, N, L, mask, h, w = self.data.load(objid)
            self.I = I
            self.N = N
            self.L = L
            self.M = mask.reshape(-1, h, w)
            self.h = h
            self.w = w
            self.valid_array = [1]
        else:
            raise Exception("Invalid Data Type")


    def __getitem__(self, index):

        if self.data_type == 'Cycles':
            pixelid = self.valid_array[index]
            dataid = self.data_id_array[index]
            # lightids =  np.random.permutation(self.L[dataid].shape[0])[:np.random.randint(self.min_nimg, self.max_nimg+1)]           
            # if self.representation_type == 'OBSMAP': # for observation map
            #     obsmap, obsmask, nml, light = get_data_observation_map(self.blocksize, self.I[dataid][index,:,idxData], self.N[dataid][index,:].reshape(3,1), light, self.thetas)
            #     return obsmap, obsmask, nml, index.astype(np.int32), setid
            # elif self.representation_type == 'RAW_PATCH'  and self.data_split == 1:

            normalize_method = 'max'
            if np.max(obs) > 0 and normalize_method == 'max':
                temp = np.mean(obs, axis=0)
                obs /= (np.max(temp[temp>0]) + 1.0e-6)
            obs, nml, mask = get_data_patch(self.block_size, self.I[dataid][:,:,lightids], self.N[dataid], self.M[dataid], self.h[dataid], self.w[dataid], pixelid)
            light = self.L[dataid][lightids,:].transpose(1,0)

            return obs, nml, mask, light
        if self.data_type == 'DiLiGenT':
            lightids =  np.random.permutation(self.L.shape[0])[:np.random.randint(self.min_nimg, self.max_nimg+1)]            
            light = self.L[lightids,:].transpose(1,0)            
            obs = self.I.reshape(self.h, self.w, self.I.shape[1], self.I.shape[2]).transpose(2, 0, 1, 3)
            obs = obs[:, :, :, lightids]
           
            normalize_method = 'max'
            if np.max(obs) > 0 and normalize_method == 'max':
                temp = np.mean(obs, axis=0)
                obs /= (np.max(temp[temp>0]) + 1.0e-6)

            nml = self.N.reshape(self.h, self.w, 3).transpose(2, 0, 1)
            mask = self.M         
            return obs, nml, mask, light



    def __len__(self):
        return len(self.valid_array)
