import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Autoencoder, self).__init__()
        self.original_input_dim = input_dim 

        encoder_layers = []
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, h))
            encoder_layers.append(nn.ReLU())
            input_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck = nn.Linear(input_dim, 4)
        input_dim = 4

        decoder_layers = []
        for h in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(input_dim, h))
            decoder_layers.append(nn.ReLU())
            input_dim = h
        decoder_layers.append(nn.Linear(input_dim, self.original_input_dim))  
        decoder_layers.append(nn.Sigmoid()) 
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        bottlenecked = self.bottleneck(encoded)
        decoded = self.decoder(bottlenecked)
        return decoded


