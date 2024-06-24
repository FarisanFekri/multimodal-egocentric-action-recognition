"""
This class is the same for the EMG or RGB features
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable

class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims, variational=True):
        super(VariationalEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dims = latent_dims
        self.variational = variational
        self.encoder = nn.Sequential(nn.Linear(self.in_channels, self.latent_dims),
                                     nn.BatchNorm1d(self.latent_dims),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(latent_dims, latent_dims),
                                     nn.BatchNorm1d(self.latent_dims),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.2),
                                     )

        self.mu = nn.Linear(latent_dims, latent_dims)
        self.sigma = nn.Linear(latent_dims, latent_dims)
    
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        if self.variational:
            return self.mu(h), self.sigma(h)
        else:
            return h
    
class Decoder(nn.Module):
    def __init__(self, latent_dims, out_channels):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.out_channels = out_channels
        self.decoder = nn.Sequential(
                             nn.Linear(self.latent_dims, self.latent_dims),
                             nn.ReLU(inplace=True),
                             nn.BatchNorm1d(self.latent_dims),
                             nn.Linear(latent_dims, self.out_channels),
                             nn.Dropout(0.2),
                             )
    
    def forward(self, z):
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dims, out_channels, variational=True, resume_from = None):
        super(VariationalAutoencoder, self).__init__()
        self.variational = variational
        self.encoder = VariationalEncoder(in_channels, latent_dims, variational=self.variational)
        self.decoder = Decoder(latent_dims, out_channels)
        self.resume_from = resume_from
        
    
    def load_on(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 *logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        if self.variational:
            mu, log_var = self.encoder(x)
            z = self.reparametrize(mu, log_var)
            res = self.decoder(z)
            return res, z, mu, log_var
        else:
            z = self.encoder(x)
            res = self.decoder(z)
            return res, z
        
    def load_last_model(self):
        encoder_state_dict = torch.load(self.resume_from['encoder'])['encoder']
        decoder_state_dict = torch.load(self.resume_from['decoder'])['decoder']
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)