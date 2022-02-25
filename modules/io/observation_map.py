import numpy as np

def rot2d_np(x, theta): # input shape [3, numelement]
    x_ = np.zeros(x.shape, np.float32)
    rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    temp = rotmat @ x[:2,:]
    x_ = x.copy()
    x_[:2,:] = temp
    return x_

def normalize_obs_map(obs, mask, obs_w, method):
    # normalize  obs map

    if method == 'mean':
        temp = np.sum(np.mean(obs.reshape(-1, 3, obs_w * obs_w), 1), 1)
        temp = np.tile(temp.reshape(temp.shape[0], 1, 1, 1), (1, 3, obs_w, obs_w))  + 1.0e-6
        obs = obs / temp
        scale1  = temp
        temp = np.sum(mask.reshape(-1, obs_w * obs_w),1)
        temp = np.tile(temp.reshape(-1, 1, 1, 1), (1, 3, obs_w, obs_w))
        obs = obs * temp
        scale = temp / scale1
    elif method == 'max':
        scale = np.max(np.mean(obs.reshape(-1, 3, obs_w * obs_w), 1), 1)
        scale = np.tile(scale.reshape(scale.shape[0], 1, 1, 1), (1, 3, obs_w, obs_w))  + 1.0e-6
        obs = obs / scale
    else:
        print('method must be from [mean, max]', file=sys.stderr)

    return obs, scale

def vec2obsid(x, obs_w):

    if len(x.shape) == 2:
        p = x[0,:] * 0.5 * obs_w + 0.5 * obs_w;
        q = x[1,:] * 0.5 * obs_w + 0.5 * obs_w;
        return q.astype(np.int32) * obs_w + p.astype(np.int32)
    if len(x.shape) == 3:
        p = x[:,0,:] * 0.5 * obs_w + 0.5 * obs_w;
        q = x[:,1,:] * 0.5 * obs_w + 0.5 * obs_w;
        return q.astype(np.int32) * obs_w + p.astype(np.int32)

def gen_observation_map(obs_w, obs, light, idxData = []):

    num_rotation_angles = light.shape[0]
    obsmap = np.zeros((num_rotation_angles, 3, obs_w * obs_w), np.float32)
    obsmask = np.zeros((num_rotation_angles, 1, obs_w * obs_w), np.float32)
    light_idx = vec2obsid(light, obs_w)
    for k in range(num_rotation_angles):
        obsmap[k,:,light_idx[k,:]] = obs#.transpose(1,0)
        obsmask[k,:,light_idx[k,:]] = 1
    obsmap = obsmap.reshape(-1, 3, obs_w, obs_w)
    obsmask = obsmask.reshape(-1, 1, obs_w, obs_w)
    obsmap, scale = normalize_obs_map(obsmap, obsmask, obs_w, 'max')
    return obsmap, obsmask

def get_data_observation_map(obs_w, obs, nml, light, thetas):
    light_r = np.array([rot2d_np(light.transpose(1,0), thetas[k]) for k in range(len(thetas))])
    nml_r = np.array([rot2d_np(nml, thetas[k]) for k in range(len(thetas))])
    obsmap, obsmask = gen_observation_map(obs_w, obs, light_r)
    return obsmap, obsmask, nml_r, light_r
