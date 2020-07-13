from __future__ import division

import torch
import torch.nn as nn

class UAP(nn.Module):
    def __init__(self,
                shape=(224, 224),
                num_channels=3,
                mean=[0.,0.,0.],
                std=[1.,1.,1.],
                use_cuda=True):
        super(UAP, self).__init__()

        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

        self.mean_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]
        if use_cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]
        if use_cuda:
            self.std_tensor = self.std_tensor.cuda()

    def forward(self, x):
        uap = self.uap
        # Put image into original form
        orig_img = x * self.std_tensor + self.mean_tensor
        
        # Add uap to input
        adv_orig_img = orig_img + uap
        # Put image into normalized form
        adv_x = (adv_orig_img - self.mean_tensor)/self.std_tensor

        return adv_x
