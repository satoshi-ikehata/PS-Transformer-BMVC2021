from . import network
import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F

def optimizer_setup(net, lr = 0.5 * 1.0e-4, init=True):
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr}]
    return net, torch.optim.Adam(optim_params, betas=(0.9, 0.999), weight_decay=0)

def print_model_parameters(model, model_name = None):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    if model_name is not None:
        print(f'{model_name} # parameters: {params}')
    else:
        print('# parameters: %d' % params)

def mode_change(net, Training):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

def angular_error(x1, x2, mask = None):

    if mask is not None:
        dot = torch.sum(x1 * x2 * mask, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        emap = torch.abs(180 * torch.acos(dot)/np.pi) * mask
        mae = torch.sum(emap) / torch.sum(mask)
        return mae
    if mask is None:
        dot = torch.sum(x1 * x2, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        error = torch.abs(180 * torch.acos(dot)/np.pi)
        return error

def loadmodel(model, filename):
    params = torch.load('%s' % filename)
    model.load_state_dict(params)
    print('Load %s' % filename)
    return model

def savemodel(model, filename):
    print('Save %s' % filename)
    torch.save(model.state_dict(), filename)


class model():
    def __init__(self, device, train_block_size = 32, test_block_size = 32, num_enc_sab = 3, lr = 0.5 * 1e-4, img_channels = 3, ln = False, attention_dropout = 0.1):
        self.device = device

        self.device = device
        self.train_block_size = train_block_size
        self.test_block_size = test_block_size

        print('model (PS-TRANSFORMER) is created!! BlockSize %d, Encoder %d Layers, ImgDepth %d, LayerNorm %d, Dropout %.2f' % (train_block_size, num_enc_sab, img_channels, ln, attention_dropout))
        self.feature_extractor = network.Encoder(img_channels + 1) # img_channels + mask -> 64 (Patch)
        self.feature_extractor = self.feature_extractor.to(device)
        [self.feature_extractor, self.optimizer_feature_extractor] = optimizer_setup(self.feature_extractor, lr = lr) # 0.5*1e-4
        print_model_parameters(self.feature_extractor, 'feature_extracter') #

        self.transformer1 = network.Transformer(dim_input = 64 + 3, num_outputs = 1, dim_output = 3, num_enc_sab = num_enc_sab, num_dec_sab = 0, num_inds=32, dim_hidden=256, num_heads=8, ln=ln, attention_dropout = attention_dropout) #12
        self.transformer1 = self.transformer1.to(self.device)
        [self.transformer1, self.optimizer_transformer1] = optimizer_setup(self.transformer1, lr = lr) # 0.5*1e-4
        print_model_parameters(self.transformer1, 'transformer1')

        self.transformer2 = network.Transformer(dim_input = img_channels + 3, num_outputs = 1, dim_output = 3, num_enc_sab = num_enc_sab, num_dec_sab = 0, num_inds=32, dim_hidden=256, num_heads=8, ln=ln, attention_dropout = attention_dropout) #Encoder
        self.transformer2= self.transformer2.to(self.device)
        [self.transformer2, self.optimizer_transformer2] = optimizer_setup(self.transformer2, lr = lr) # 0.5*1e-4
        print_model_parameters(self.transformer2, 'transformer2')

        self.decoder_single = network.DecoderRegressionFCN(64 + 3, 3) # img_channels + mask
        self.decoder_single = self.decoder_single.to(device)
        [self.decoder_single, self.optimizer_decoder_single] = optimizer_setup(self.decoder_single, lr = lr) # 0.5*1e-4
        print_model_parameters(self.decoder_single, 'decoder_single')

        self.decoder = network.DecoderRegressionFCN(512 + 1, 3) # img_channels + mask
        self.decoder = self.decoder.to(device)
        [self.decoder, self.optimizer_decoder] = optimizer_setup(self.decoder, lr = lr) # 0.5*1e-4
        print_model_parameters(self.decoder, 'decoder')

        self.criterionL2 = nn.MSELoss(reduction = 'sum').to(device)


    def scale_lr(self, scale):
        print('learning rate updated  %.5f -> %.5f' % (self.optimizer_transformer1.param_groups[0]['lr'], self.optimizer_transformer1.param_groups[0]['lr'] * scale))
        self.optimizer_transformer1.param_groups[0]['lr'] *= scale
        self.optimizer_transformer2.param_groups[0]['lr'] *= scale
        self.optimizer_feature_extractor.param_groups[0]['lr'] *= scale
        self.optimizer_decoder.param_groups[0]['lr'] *= scale
        self.optimizer_decoder_single.param_groups[0]['lr'] *= scale

    def set_mode(self, mode):
        if  mode in 'Train':
            print('PS-Transformer, TrainMode')
            self.mode = 'Train'
            mode_change(self.transformer1, True)
            mode_change(self.transformer2, True)
            mode_change(self.feature_extractor, True)
            mode_change(self.decoder, True)
            mode_change(self.decoder_single, True)
        else:
            print('PS-Transformer, TestMode')
            self.mode = 'Test'
            mode_change(self.transformer1, False)
            mode_change(self.transformer2, False)
            mode_change(self.feature_extractor, False)
            mode_change(self.decoder, False)
            mode_change(self.decoder_single, False)


    def save_models(self, dirpath):
        os.makedirs(dirpath, exist_ok = True)
        savemodel(self.transformer1, dirpath + '/transformer1.pytmodel')
        savemodel(self.transformer2, dirpath + '/transformer2.pytmodel')
        savemodel(self.feature_extractor, dirpath + '/feature_extractor.pytmodel')
        savemodel(self.decoder, dirpath + '/decoder.pytmodel')
        savemodel(self.decoder_single, dirpath + '/decoder_single.pytmodel')

    def load_models(self, dirpath):
        print(dirpath)
        self.transformer1 = loadmodel(self.transformer1, dirpath + '/transformer1.pytmodel')
        self.transformer2 = loadmodel(self.transformer2, dirpath + '/transformer2.pytmodel')
        self.feature_extractor =  loadmodel(self.feature_extractor, dirpath + '/feature_extractor.pytmodel')
        self.decoder =  loadmodel(self.decoder, dirpath + '/decoder.pytmodel')
        self.decoder_single =  loadmodel(self.decoder_single, dirpath + '/decoder_single.pytmodel')


    def step(self, batch, min_nimg = None, max_nimg = None):
        # obs [B, channel, p, p, nimg_buffer]
        # nml [B, 3, p, p]
        # mask [B, 1, p, p]
        # light [B, 3, nimg_buffer]
        # batch: obs, nml, mask, light
        obs = batch[0]
        nml = batch[1]
        mask = batch[2]
        light = batch[3]
     

        """ pixelwise branch """
        loss = 0
        B = obs.shape[0]
        img_channels = obs.shape[1]
        h = obs.shape[2]
        w = obs.shape[3]
  
        nimg_buffer = obs.shape[4]

        nml = nml.to(self.device)
        mask = mask.to(self.device)

        if min_nimg and max_nimg:
            light_idx = np.random.permutation(nimg_buffer)[:np.random.randint(min_nimg, max_nimg+1)]
            obs = obs[:, :, :, :, light_idx].to(self.device)
            light = light[:, :, light_idx].to(self.device)
        else:
            light_idx = -1
            obs = obs.to(self.device)
            light = light.to(self.device)
        numlight = light.shape[2]

        obs_t = obs.reshape(-1, img_channels, h * w, numlight).permute(0, 2, 3, 1) # [B, h * w,  numlight, img_channels]
        obs_t  = obs_t.reshape(-1, numlight, img_channels)
        light_t = light.reshape(-1, 3, numlight, 1).expand(-1, -1, -1, h * w) # [B, 3, numlight, h * w]
        light_t = light_t.permute(0, 3, 2, 1).reshape(-1, numlight, 3) # [B * h * w, numlight, 3]
        nml_t = nml.reshape(-1, 3, h * w).permute(0, 2, 1).reshape(-1, 3) # [B * h * w, 3]
        mask_t = mask.reshape(-1, 1, h * w).permute(0, 2, 1).reshape(-1, 1) # [B * h * w, 1]
        data_t = torch.cat([obs_t, light_t], dim=2) # [B * h * w, numlight, 3 + img_channels]
        nout_t, feats_t = self.transformer2(data_t)
        nout_t = F.normalize(nout_t, p=2, dim=1)

        loss_t = self.criterionL2(nml_t * mask_t, nout_t * mask_t) / torch.sum(mask_t)
        loss += loss_t
        feats_t = feats_t.reshape(-1, h * w, feats_t.shape[1]).permute(0, 2, 1) # [B, dimFeat, h * w]

        """ imagewise branch """
        feats_cnn = []
        loss_single = 0
        for k in range(numlight):
            obs_k = obs[:, :, :, :, k]
            feat_cnn = self.feature_extractor(torch.cat([obs_k, mask], dim=1))
            nout_cnn_single = F.normalize(self.decoder_single(torch.cat([feat_cnn, light[:, :, k].reshape(-1, 3, 1, 1).expand(-1, -1, h, w)], dim=1)), p=2, dim=1)
            nout_cnn_single = nout_cnn_single.reshape(-1, 3, h * w).permute(0, 2, 1).reshape(-1, 3)
            loss_single += self.criterionL2(nml_t * mask_t, nout_cnn_single * mask_t) / torch.sum(mask_t)
            feats_cnn.append(feat_cnn)
        loss += (loss_single / numlight)

        feats_cnn = torch.stack(feats_cnn, dim = 1) # [B, numLight, outdim, h, w]
        feats_cnn = feats_cnn.reshape(feats_cnn.shape[0], feats_cnn.shape[1], feats_cnn.shape[2], h * w)
        feats_cnn = torch.cat([feats_cnn, light.permute(0,2,1).reshape(light.shape[0], light.shape[2], light.shape[1], 1).expand(-1, -1, -1, feats_cnn.shape[3])], dim=2)# feats_cnn [B, numLight, outdim + 3, p * p]

        feats_cnn = feats_cnn.permute(0, 3, 1, 2).reshape(-1, feats_cnn.shape[1], feats_cnn.shape[2]) # [B * h * w, numLight, numFeat]
        nout_cnn, feats_cnn = self.transformer1(feats_cnn)
        nout_cnn = F.normalize(nout_cnn, p=2, dim=1)

        loss_cnn = self.criterionL2(nml_t * mask_t, nout_cnn * mask_t) / torch.sum(mask_t)
        loss += loss_cnn

        """ Fusion """
        feats_t = feats_t.reshape(-1, feats_t.shape[1], h, w)
        feats_cnn = feats_cnn.permute(1,0).reshape(-1, feats_cnn.shape[1], h, w)

        feats =torch.cat([feats_t, feats_cnn, mask], dim=1)
        npatch = F.normalize(self.decoder(feats), p=2, dim=1) * mask

        npatch_true = nml.reshape(-1, 3, h, w) * mask
        loss_nml = self.criterionL2(npatch, npatch_true) / torch.sum(mask)
        loss += loss_nml
        error = angular_error(npatch.cpu(), npatch_true.cpu(), mask = mask.cpu())


        output = (127.0 * (1 + torch.cat([npatch * mask, npatch_true * mask], 2))).to(torch.uint8)

        if self.mode in 'Train':
            self.optimizer_feature_extractor.zero_grad()
            self.optimizer_transformer1.zero_grad()
            self.optimizer_transformer2.zero_grad()
            self.optimizer_decoder.zero_grad()
            self.optimizer_decoder_single.zero_grad()
            loss.backward()
            self.optimizer_feature_extractor.step()
            self.optimizer_transformer1.step()
            self.optimizer_transformer2.step()
            self.optimizer_decoder.step()
            self.optimizer_decoder_single.step()

        return output.cpu().detach(), loss.cpu().detach(), error.cpu().detach()
