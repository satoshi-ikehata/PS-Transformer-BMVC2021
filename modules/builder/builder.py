import torch
from tqdm import tqdm

from modules.model import pstransformer

class builder():
    def __init__(self, device):
        self.device = device
        self.net = pstransformer.model(device)
        self.global_iter = 0

    def run(self, mode, data = None, batch_size = 8):

        if mode == 'Train':
            self.net.set_mode('Train')
            train_data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=True, num_workers=0, pin_memory=True)
            losses = 0
            errors = 0
            for batch in tqdm(train_data_loader, leave=False):
                output, loss, error  = self.net.step(batch) # output = [B, 3, h, w]
                losses += loss
                errors += error
                self.global_iter += 1
            return losses/len(train_data_loader), errors/len(train_data_loader), output


        if mode == 'Test':
            self.net.set_mode('Test')
            test_data_loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle=False, num_workers=0, pin_memory=True)
            for batch in test_data_loader:
                output, loss, error  = self.net.step(batch) # output = [B, 3, h, w]
            return output, error
