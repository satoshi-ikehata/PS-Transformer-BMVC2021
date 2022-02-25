import glob
import os,sys
import cv2
import numpy as np
import math
import scipy.io
from numpy import unique

class io():
    def __init__(self, data_root):

        self.data_root = data_root
        self.img_channels = 3
        self.datatype = 'DiLiGenT'
        self.ext = 'PNG'
        self.objlists = []
        data_path = data_root
        objlist = []
        [objlist.append(p) for p in glob.glob(data_path + '/*%s' % self.ext, recursive=True) if os.path.isdir(p)]
        objlist = sorted(objlist)
        self.objlist = objlist

    def normalize(self, imgs): # [NLight, H, W ,C]
        h, w, c = imgs[0].shape
        imgs = [img.reshape(-1, 1) for img in imgs]
        img = np.hstack(imgs)
        norm = np.sqrt((img * img).clip(0.0).sum(1))
        img = img / (norm.reshape(-1,1) + 1e-10)
        imgs = np.split(img, img.shape[1], axis=1)
        imgs = [img.reshape(h, w, -1) for img in imgs]
        return imgs

    def get_num_object(self):
        return len(self.objlist)

    def get_num_set(self):
        return 1

    def load(self, objid, view=-1, margin = 0, imgsize=256):
        objlist = self.objlist
        objid = objid

        objname = objlist[objid].split('/')[-1]
        directlist = []

        print(f'Loading {objname} (DiLiGenT)')

        if view == -1:
            [directlist.append(p) for p in glob.glob(objlist[objid] + '/0*.png',recursive=True) if os.path.isfile(p)]
        else:
            [directlist.append(p) for p in glob.glob(objlist[objid] + '/view_%02d/0*.png' % (view+1),recursive=True) if os.path.isfile(p)]

        directlist = sorted(directlist)
        I = []

        if os.name == 'posix':
            temp = directlist[0].split("/")
        if os.name == 'nt':
            temp = directlist[0].split("\\")
        img_dir = "/".join(temp[:-1])

        if os.name == 'posix':
            mask = cv2.cvtColor(cv2.imread(img_dir + '/mask.png', flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
            mask = np.float32(mask)/mask.max()
            if len(mask.shape) == 3:
                mask = (np.mean(mask, axis=2) > 0).astype(np.float32)
        if os.name == 'nt':
            mask = cv2.imread(img_dir + '/mask.png', flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = np.float32(mask)/mask.max()
            if len(mask.shape) == 3:
                mask = (np.mean(mask, axis=2) > 0).astype(np.float32)


        rows, cols = np.nonzero(mask)
        bh = np.max(rows)-np.min(rows)
        bw = np.max(cols)-np.min(cols)
        if bh > bw:
            r0 = np.min(rows)
            r1 = np.max(rows)
            c0 = int(0.5 *(np.max(cols)+np.min(cols)) - 0.5 * bh)
            c1 = c0 + bh
        else:
            c0 = np.min(cols)
            c1 = np.max(cols)
            r0 = int(0.5 * (np.max(rows)+np.min(rows)) - 0.5 * bw)
            r1 = r0 + bw


        margin = 5
        mask = mask[r0-margin:r1+margin, c0-margin:c1+margin]
        mask = cv2.resize(mask, (imgsize, imgsize), interpolation=cv2.INTER_NEAREST)

        intensity = np.loadtxt(img_dir + '/light_intensities.txt', np.float32) # N x 3
        for i, img_path in enumerate(directlist):
            if objid == 1 and i < 20:
                continue;
            img = cv2.cvtColor(cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
            img = np.float32(img)/65535.0
            img[:,:,0] = img[:,:,0] / intensity[i,0]
            img[:,:,1] = img[:,:,1] / intensity[i,1]
            img[:,:,2] = img[:,:,2] / intensity[i,2]
            img = img[r0-margin:r1+margin, c0-margin:c1+margin,:]
            img = cv2.resize(img,  (imgsize, imgsize), interpolation=cv2.INTER_NEAREST)
            I.append(img)
            h = img.shape[0]
            w = img.shape[1]
       
        normalize=True
        if normalize == True:
            I = self.normalize(I)
        I = np.array(I)

        nml_path = img_dir + '/Normal_gt.mat'
        if os.path.isfile(nml_path):
            mat = scipy.io.loadmat(nml_path)
            nml = np.array(mat['Normal_gt'], np.float32)
            N = nml[r0-margin:r1+margin, c0-margin:c1+margin, :]
            N = cv2.resize(N,  (imgsize, imgsize), interpolation=cv2.INTER_NEAREST)
            N = np.reshape(N, (h * w, 3))

        L = np.loadtxt(img_dir + '/light_directions.txt', np.float32) # N x 3
        if objid == 1:
            L = L[20:,:]

        I = np.reshape(I, (-1, h * w, 3))
        I = np.transpose(I, (1,2,0))
        return  I, N, L, mask, h, w
