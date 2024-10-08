# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:12:32 2024

pytorch implementation from
https://github.com/siddinc/eeg2vec

@author: jstieger
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter

class ResidualBlockA(nn.Module):
    def __init__(self, n_filters, kernel_size, dilation_rate):
        super(ResidualBlockA, self).__init__()
        self.tanh_conv = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=dilation_rate, padding=0)
        self.sigm_conv = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=dilation_rate, padding=0)
        self.skip_conv = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=1)

    def forward(self, x):
        padding = (self.tanh_conv.kernel_size[0] - 1) * self.tanh_conv.dilation[0]
        tanh_out = torch.tanh(self.tanh_conv(F.pad(x, (padding, 0))))
        sigm_out = torch.sigmoid(self.sigm_conv(F.pad(x, (padding, 0))))
        z = tanh_out * sigm_out
        skip = self.skip_conv(z)
        res = skip + x
        return res, skip

class ResidualBlockB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlockB, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.shortcut_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same')
        self.shortcut_bn = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x_short = self.shortcut_bn(self.shortcut_conv(x))
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        
        x_add = x_short + x
        x_out = F.leaky_relu(x_add, negative_slope=0.2)
        return x_out

class Autoencoder(nn.Module):
    def __init__(self, input_shape, n_filters, dilation_depth,
                 output_shape, kernel_size, class_weights):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_filters = n_filters
        self.dilation_depth = dilation_depth
        self.class_weights = class_weights

        self.initial_conv = nn.Sequential( 
            nn.Conv1d(in_channels=1, out_channels=n_filters,
                      kernel_size=2, dilation=1, padding=0),
            nn.Dropout(0.25))

        self.residual_blocks = nn.ModuleList([
            ResidualBlockA(n_filters, kernel_size, kernel_size ** i) 
            for i in range(1, dilation_depth + 1)])
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=n_filters, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.25),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.25),
        )

        self.flatten = nn.Flatten()
        self.encoder_dense = nn.Linear(in_features=self.calculate_flatten_size(), out_features=output_shape[0])

        self.decoder_dense = nn.Linear(in_features=output_shape[0], out_features=self.calculate_flatten_size())
        self.decoder_residual_blocks = nn.Sequential(
            nn.Dropout(p=0.25),
            ResidualBlockB(64, 64, 3),
            ResidualBlockB(64, 32, 3),
            nn.Dropout(p=0.25),
            ResidualBlockB(32, 16, 3),
        )
        self.output_conv = nn.Sequential( 
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=0),
            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(output_shape[0] + 7, output_shape[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.25),
            nn.Linear(output_shape[0],output_shape[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.25),
            nn.Linear(output_shape[0],len(class_weights))
            )
        
        #butterworth filter for HFB estimation
        samp_freq = 1000.0
        order = 5
        lowcut = 60
        highcut = 180
        signal_length = input_shape[0]
        freqs = torch.fft.fftfreq(signal_length, d=1/samp_freq)
        H = 1 / (1 + ((freqs / lowcut) ** (2 * order))) * \
            (1 / (1 + ((highcut / freqs) ** (2 * order))))
        self.butter_filt = H.to(class_weights.device)
        

    def calculate_flatten_size(self):
        dummy_input = torch.zeros(1, 1, self.input_shape[0])
        x = self.initial_conv(F.pad(dummy_input, (1, 0)))
        for block in self.residual_blocks:
            x, _ = block(x)
        x = self.conv_layers(x)
        return x.numel()
    
    def encode(self, x):
        x = x.unsqueeze(1)  # Reshape input to (batch, channels, length)
        x = self.initial_conv(F.pad(x, (1, 0)))
        skips = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skips.append(skip)
        x = torch.sum(torch.stack(skips), dim=0)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_layers(x)
        x = self.flatten(x)
        encoded = self.encoder_dense(x)
        return encoded
    
    def decode(self, x):
        
        x = self.decoder_dense(x)
        x = x.view(-1, 64, x.shape[-1] // 64)
        x = self.decoder_residual_blocks(x)
        x = torch.tanh(self.output_conv(x)).squeeze()
        
        return x
    
    def classify(self, latent_data, batch):
        
        x = torch.cat((latent_data,batch['dist'],batch['loc']), dim=-1)
        logits = self.classifier(x)
        
        return logits
    
    
    def forward(self, batch):
        x = batch['wave_source']
        latent_data = self.encode(x)
        recon_x = self.decode(latent_data)
        class_logits = self.classify(latent_data, batch)
        #return to dictionary
        batch['latent_data'] = latent_data
        batch['recon_x'] = recon_x
        batch['class_logits'] = class_logits.squeeze()
        
        return batch
    
    def loss_function(self, batch,
                      weight_lambda = 1e-5,
                      recon_lambda = 100,
                      class_lambda = 1):
        
        #reconstruction loss - raw points
        RECON_WAVE = recon_lambda*F.mse_loss(batch['recon_x'],
                                batch['wave_target'], reduction='mean')
        
        
        '''
        #use derivitive error to model high frequency components and slopes
        recon_x_slope = torch.diff(batch['recon_x'])
        wave_slope = torch.diff(batch['wave'])
        RECON_SLOPE = recon_lambda*F.mse_loss(recon_x_slope,
                                wave_slope, reduction='mean')
        
        recon_x_curve = torch.diff(recon_x_slope)
        wave_curve = torch.diff(wave_slope)
        RECON_CURVE = recon_lambda*F.mse_loss(recon_x_curve,
                                wave_curve, reduction='mean')
        '''
        
        '''
        #reconstruction loss HFB power
        #convert to frequency domain
        fft_recon_x = torch.fft.fft(batch['recon_x'])
        fft_x = torch.fft.fft(batch['wave'])
        #filter
        fft_x_filt = fft_x*self.butter_filt
        fft_recon_x_filt = fft_recon_x*self.butter_filt
        #get power
        #x_HFB = torch.real(torch.fft.ifft(fft_x_filt))**2
        #recon_x_HFB = torch.real(torch.fft.ifft(fft_recon_x_filt))**2
        #get bandpass
        x_HFB = torch.real(torch.fft.ifft(fft_x_filt))
        recon_x_HFB = torch.real(torch.fft.ifft(fft_recon_x_filt))
        
        RECON_HFB = recon_HFB*F.mse_loss(x_HFB[:,10:-10],
                                recon_x_HFB[:,10:-10], reduction='mean')
        '''
        
        if torch.isnan(RECON_WAVE):
            print("RECON_WAVE loss is NaN, replacing with zero.")
            RECON_WAVE = torch.tensor(200.0)
        
        '''
        #reparameterization loss
        KLD = -0.5 * torch.sum(1 + batch['logvar'] - 
                               batch['mu'].pow(2) - 
                               batch['logvar'].exp())
        '''

        l2_loss = l2_loss = sum(p.pow(2).sum() 
                                for name, p in self.named_parameters())
              
        WEIGHT_LOSS = weight_lambda*(l2_loss)
        
        #calculate the classification loss for valid
        CLASS_LOSS = class_lambda*F.cross_entropy(
            batch['class_logits'],
            batch['class_label'],
            weight=self.class_weights,
            reduction='mean')
        
        if torch.isnan(CLASS_LOSS):
            raise ValueError('Nan Batch')
            print("Classification loss is NaN, replacing with zero.")
            CLASS_LOSS = torch.tensor(200.0)
        
        #claculate classification accuracy
        predicted_class = torch.argmax(
            batch['class_logits'], dim = -1).long()
        correct_predictions = (predicted_class == batch['class_label'])
        num_correct = torch.sum(correct_predictions).cpu().item()
        CLASS_ACCURACY = (num_correct ,len(correct_predictions))
        
       
        #TOTAL_LOSS = RECON_WAVE + RECON_FFT +  KLD + WEIGHT_LOSS
        TOTAL_LOSS = RECON_WAVE + WEIGHT_LOSS + CLASS_LOSS
            
        
        
        loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                      'RECON_WAVE': RECON_WAVE.cpu().item(),
                      'WEIGHTS': WEIGHT_LOSS.cpu().item(),
                      'CLASS_LOSS': CLASS_LOSS.cpu().item(),
                      'CLASS_ACCURACY': CLASS_ACCURACY}
        
        return TOTAL_LOSS, loss_dict    



class PureAutoEncoder(nn.Module):
    def __init__(self, input_shape, n_filters, dilation_depth,
                 output_shape, kernel_size):
        super(PureAutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_filters = n_filters
        self.dilation_depth = dilation_depth

        self.initial_conv = nn.Sequential( 
            nn.Conv1d(in_channels=1, out_channels=n_filters,
                      kernel_size=2, dilation=1, padding=0),
            nn.Dropout(0.25))

        self.residual_blocks = nn.ModuleList([
            ResidualBlockA(n_filters, kernel_size, kernel_size ** i) 
            for i in range(1, dilation_depth + 1)])
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=n_filters, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.25),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.25),
        )

        self.flatten = nn.Flatten()
        self.encoder_dense = nn.Linear(in_features=self.calculate_flatten_size(), out_features=output_shape[0])

        self.decoder_dense = nn.Linear(in_features=output_shape[0], out_features=self.calculate_flatten_size())
        self.decoder_residual_blocks = nn.Sequential(
            nn.Dropout(p=0.25),
            ResidualBlockB(64, 64, 3),
            ResidualBlockB(64, 32, 3),
            nn.Dropout(p=0.25),
            ResidualBlockB(32, 16, 3),
        )
        self.output_conv = nn.Sequential( 
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=0),
            )
    
    def calculate_flatten_size(self):
        dummy_input = torch.zeros(1, 1, self.input_shape[0])
        x = self.initial_conv(F.pad(dummy_input, (1, 0)))
        for block in self.residual_blocks:
            x, _ = block(x)
        x = self.conv_layers(x)
        return x.numel()
    
    def encode(self, x):
        x = x.unsqueeze(1)  # Reshape input to (batch, channels, length)
        x = self.initial_conv(F.pad(x, (1, 0)))
        skips = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skips.append(skip)
        x = torch.sum(torch.stack(skips), dim=0)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_layers(x)
        x = self.flatten(x)
        encoded = self.encoder_dense(x)
        return encoded
    
    def decode(self, x):
        
        x = self.decoder_dense(x)
        x = x.view(-1, 64, x.shape[-1] // 64)
        x = self.decoder_residual_blocks(x)
        x = torch.tanh(self.output_conv(x)).squeeze()
        
        return x
    
    def classify(self, latent_data, batch):
        
        x = torch.cat((latent_data,batch['dist'],batch['loc']), dim=-1)
        logits = self.classifier(x)
        
        return logits
    
    
    def forward(self, batch):
        x = batch['wave_source']
        latent_data = self.encode(x)
        recon_x = self.decode(latent_data)
        #class_logits = self.classify(latent_data, batch)
        #return to dictionary
        batch['latent_data'] = latent_data
        batch['recon_x'] = recon_x
        #batch['class_logits'] = class_logits.squeeze()
        
        return batch
    
    def loss_function(self, batch,
                      weight_lambda = 1e-4,
                      recon_lambda = 100):
        
        #reconstruction loss - raw points
        RECON_WAVE = recon_lambda*F.mse_loss(batch['recon_x'],
                                batch['wave_target'], reduction='mean')

        l2_loss = l2_loss = sum(p.pow(2).sum() 
                                for name, p in self.named_parameters())
              
        WEIGHT_LOSS = weight_lambda*(l2_loss)
        
        TOTAL_LOSS = RECON_WAVE + WEIGHT_LOSS 
            
        
        loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                      'RECON_WAVE': RECON_WAVE.cpu().item(),
                      'WEIGHTS': WEIGHT_LOSS.cpu().item()}
        
        return TOTAL_LOSS, loss_dict  