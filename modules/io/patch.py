import numpy as np

def crop_index(u, v, w, h, psize):
    p = psize//2
    urange = range(u - p + 1, u + p + 1)
    vrange = range(v - p + 1, v + p + 1)
    uu, vv = np.meshgrid(urange, vrange)
    valid = np.nonzero((uu >= 0) * (uu < w) * (vv >= 0) * (vv < h))
    return vec2ind(uu[valid], vv[valid], w, h).flatten(), vec2ind(valid[1], valid[0], psize, psize).flatten(), vec2ind(psize // 2 - 1, psize // 2 - 1, psize, psize)

def ind2vec(ind, w, h): # u, v [0, w-1];[0,h-1]
    v = ind // w
    u = ind - v * w
    return u, v

def vec2ind(u, v, w, h):
    return v * w + u

def get_data_patch(p, obs, nml, mask, h, w, index):
    u, v = ind2vec(index, w, h)
    index_cropped, nonzero_idx, center_idx = crop_index(u, v, w, h, p)
    patch = np.zeros((p * p, 3, obs.shape[2]), np.float32)
    m = np.zeros((p * p, 1), np.float32)
    patch[nonzero_idx, :, :] = obs[index_cropped, :, :] # [numcrop, 3, numLight]
    m[nonzero_idx, :] = mask[index_cropped,:]
    n = np.zeros((p * p, 3), np.float32)
    n[nonzero_idx, :] = nml[index_cropped, :] # [3, numcrop]

    patch = np.reshape(patch, (p, p, 3, obs.shape[2])).transpose(2,0,1,3)
    n = np.reshape(n, (p, p, 3))
    m = np.reshape(m, (p, p, 1))
    m = m.transpose(2,0,1)
    n = n.transpose(2,0,1)
    return patch, n, m
