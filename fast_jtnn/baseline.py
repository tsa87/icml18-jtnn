import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from nnutils import create_var, create_onehot, log_standard_categorical

class NNVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, y_size, alpha):
        super(NNVAE, self).__init__()
        self.y_size = y_size
        self.alpha = alpha
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_size),
            nn.Softmax()
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(hidden_size, latent_size)
        self.z_var = nn.Linear(hidden_size, latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size+y_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        
        self.reconstruction_loss = nn.MSELoss()
        self.prediction_loss = nn.CrossEntropyLoss()
        
    def rsample(self, h):
        batch_size = h.size(0)
        z_mean = self.z_mean(h)
        z_log_var = -torch.abs(self.z_var(h))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss 
        
    
    def decode(self, z, y):
        latent = torch.cat((z, y), 1)
        x_hat = self.decoder(latent)
        return x_hat
    
    
    def compute_reconstruction_loss(self, x, z, y):
        x_hat = self.decode(z, y)
        return self.reconstruction_loss(x_hat, x)
        

    def forward(self, x, y, beta):
        is_labeled = False if y is None else True
        
        pred, target = None, None
        loss, pred_loss, pred_acc = 0, 0, 0
        
        x = create_var(x)
        y = create_var(y)
        
        h = self.encoder(x)  
        y_hat = self.classifier(x)
        z, kl_div = self.rsample(h)

        logy = torch.mean(log_standard_categorical(y_hat))
        loss = logy + kl_div

        if is_labeled:
            target = torch.argmax(y, axis=1)
            pred = torch.argmax(y_hat, axis=1)
            
            pred_loss = self.prediction_loss(y_hat, target.long()) * self.alpha
            pred_acc = float((pred == target).sum().item()) / y.size(0)
            recon_loss = self.compute_reconstruction_loss(x, z, y)
            loss += pred_loss + recon_loss
        else:
            for i in range(self.y_size):
                y = create_onehot(len(x), self.y_size, i)
                recon_loss = self.compute_reconstruction_loss(x, z, y) 
                loss += recon_loss * torch.mean(y_hat[:, i])     
            
            y_hat_entropy = torch.sum(y_hat * torch.log(y_hat + 1e-8))
            loss += y_hat_entropy
            
        return loss, pred_loss, pred_acc, (pred, target)