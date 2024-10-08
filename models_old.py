# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:30:52 2024

module contains the model classes for the autoencoder project

@author: jstieger
"""


import torch
import torch.nn as nn
import torch.nn.functional as F





class EncoderDownSample(nn.Module):
    def __init__(self, in_feat, feed_forward,
                 max_seq_len = 80, downsample = True):
        
        super(EncoderDownSample, self).__init__()
        
        self.encoder_layer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model = in_feat,
                                       nhead = 8,
                                       dim_feedforward=feed_forward,
                                       dropout= 0.1,
                                       batch_first = True),
            nn.TransformerEncoderLayer(d_model = in_feat,
                                       nhead = 8,
                                       dim_feedforward=feed_forward,
                                       dropout= 0.1,
                                       batch_first = True),
            nn.TransformerEncoderLayer(d_model = in_feat,
                                       nhead = 8,
                                       dim_feedforward=feed_forward,
                                       dropout= 0.1,
                                       batch_first = True))
        
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, max_seq_len, in_feat))
        nn.init.normal_(self.positional_embeddings, mean=0, std=0.1)
        
        self.downsample = downsample
        if self.downsample:
            self.max_pool = nn.MaxPool1d(2)
        
        
    def forward(self, x):
        #prepare for transformer encoder: batch, seq, feature
        x = x.transpose(-2,-1)
        seq_len = x.size(1)
        pos_emb = self.positional_embeddings[:, :seq_len, :].to(x.device)
        x = x + pos_emb
        x = self.encoder_layer(x)
        #prepare for max pool
        x = x.transpose(-2,-1)
        if self.downsample:
            x = self.max_pool(x)
        return x
    

class EncoderUpSample(nn.Module):
    def __init__(self, in_feat, feed_forward,
                 max_seq_len = 80, upsample = True):
        
        super(EncoderUpSample, self).__init__()
        
        self.encoder_layer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model = in_feat,
                                       nhead = 8,
                                       dim_feedforward=feed_forward,
                                       dropout= 0.1,
                                       batch_first = True),
            nn.TransformerEncoderLayer(d_model = in_feat,
                                       nhead = 8,
                                       dim_feedforward=feed_forward,
                                       dropout= 0.1,
                                       batch_first = True),
            nn.TransformerEncoderLayer(d_model = in_feat,
                                       nhead = 8,
                                       dim_feedforward=feed_forward,
                                       dropout= 0.1,
                                       batch_first = True))
        
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, max_seq_len, in_feat))
        nn.init.normal_(self.positional_embeddings, mean=0, std=0.1)
        
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer = nn.Sequential(
                nn.ConvTranspose1d(in_channels=in_feat, out_channels=in_feat,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(in_feat),
                nn.ReLU())
        
        
    def forward(self, x):
        #prepare for transformer encoder: batch, seq, feature
        x = x.transpose(-2,-1)
        seq_len = x.size(1)
        pos_emb = self.positional_embeddings[:, :seq_len, :].to(x.device)
        x = x + pos_emb
        x = self.encoder_layer(x)
        #prepare for max pool
        x = x.transpose(-2,-1)
        if self.upsample:
            x = self.upsample_layer(x)
        return x


class InceptionHead(nn.Module):
    def __init__(self, num_feat, final_feat):
        """
        Inception network inspired head for feature extraction
        uses residual connections to extract hierarcical temporal features

        finally reduces the temporal space by half        

        Returns
        -------
        None.

        """
        
        super(InceptionHead, self).__init__()
        
        #divide hierarchically
        head_feat = int(num_feat/4)
        
        #nine block
        self.nine_layers = nn.ModuleList([ 
            nn.Conv1d(1, head_feat, kernel_size=9, dilation=1, padding=4),
            nn.Conv1d(1, head_feat, kernel_size=9, dilation=2, padding=8),
            nn.Conv1d(1, head_feat, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(1, head_feat, kernel_size=9, dilation=8, padding=32)])
        self.nine_bn = nn.BatchNorm1d(num_feat)
        
        #seven block
        self.seven_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=1, padding=3),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=2, padding=6),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=4, padding=12),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=8, padding=24)])
        self.seven_bn = nn.BatchNorm1d(num_feat)
        
        #five block
        self.five_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=1, padding=2),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=2, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=4, padding=8),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=8, padding=16)])
        self.five_bn = nn.BatchNorm1d(num_feat)
        
        #three block
        self.three_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=8, padding=8)])
        self.three_bn = nn.BatchNorm1d(num_feat)
        
        #down sample
        self.out_funnel = nn.Sequential( 
            nn.MaxPool1d(2), 
            nn.Conv1d(num_feat, num_feat*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_feat*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_feat*2, final_feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(final_feat),
            nn.ReLU(),
            nn.MaxPool1d(2))
            
        
    def forward(self,x):
        """
        forward method for the inception head module

        Parameters
        ----------
        x : Tensor
            input batch data of size nobs, 1, ntime.
            ntime typically 500

        Returns
        -------
        x : Tensor
            output following matrix multiplication
            87k parameters
            output batch data: nobs, 64 chan, 250 timepoints

        """
        #prep for convolutions
        x = x.unsqueeze(1)
        
        #nine block
        x = torch.cat([layer(x) for layer in self.nine_layers], dim=1)
        x = self.nine_bn(x)
        x = F.relu(x)
        resid = x
        
        #seven block
        x = torch.cat([layer(x) for layer in self.seven_layers], dim=1)
        x = self.seven_bn(x)
        x = F.relu(x)
        
        #five block
        x = torch.cat([layer(x) for layer in self.five_layers], dim=1)
        x = self.five_bn(x)
        x = F.relu(x + resid)
        resid = x
        
        #three block
        x = torch.cat([layer(x) for layer in self.three_layers], dim=1)
        x = self.three_bn(x)
        x = F.relu(x)
        
        #output block
        x = self.out_funnel(x)
        return x
    
    
class InceptionTail(nn.Module):
    def __init__(self, num_feat):
        """
        Inception network inspired head for feature extraction
        uses residual connections to extract hierarcical temporal features

        finally reduces the temporal space by half        

        Returns
        -------
        None.

        """

        super(InceptionTail, self).__init__()

        #build the initial upsampler
        self.up_sampler = nn.Sequential(
            nn.Conv1d(in_channels=num_feat, out_channels=num_feat*4,
                               kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(num_feat*4),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_feat*4, out_channels=num_feat*2,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_feat*2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_feat*2, out_channels=num_feat,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_feat, out_channels=num_feat,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU()
            )

        #divide hierarchically
        head_feat = int(num_feat/4)

        #three block
        self.three_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=8, padding=8)])
        self.three_bn = nn.BatchNorm1d(num_feat)

        #five block
        self.five_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=1, padding=2),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=2, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=4, padding=8),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=8, padding=16)])
        self.five_bn = nn.BatchNorm1d(num_feat)

        #seven block
        self.seven_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=1, padding=3),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=2, padding=6),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=4, padding=12),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=8, padding=24)])
        self.seven_bn = nn.BatchNorm1d(num_feat)

        #nine block
        self.nine_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=1, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=2, padding=8),
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=8, padding=32)])
        self.nine_bn = nn.BatchNorm1d(num_feat)

        #out block
        self.out_conv = nn.Sequential( 
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1)
            )


    def forward(self,x):
        """
        forward method for the inception head module

        Parameters
        ----------
        x : Tensor
            input batch data of size nobs, 1, ntime.
            ntime typically 500

        Returns
        -------
        x : Tensor
            output following matrix multiplication
            87k parameters
            output batch data: nobs, 64 chan, 250 timepoints

        """

        #upsample to full scale
        x = self.up_sampler(x)

        #maintain gradient
        resid = x

        #three block
        x = torch.cat([layer(x) for layer in self.three_layers], dim=1)
        x = self.three_bn(x)
        x = F.relu(x)

        #five block
        x = torch.cat([layer(x) for layer in self.five_layers], dim=1)
        x = self.five_bn(x)
        x = F.relu(x + resid)
        resid = x

        #seven block
        x = torch.cat([layer(x) for layer in self.seven_layers], dim=1)
        x = self.seven_bn(x)
        x = F.relu(x)

        #nine block
        x = torch.cat([layer(x) for layer in self.nine_layers], dim=1)
        x = self.nine_bn(x)
        x = F.relu(x + resid)

        #output block
        x = self.out_conv(x)
        x = torch.tanh(x.squeeze())
        return x
    
class InceptionTailRev(nn.Module):
    def __init__(self, num_feat):
        """
        Inception network inspired head for feature extraction
        uses residual connections to extract hierarcical temporal features

        finally reduces the temporal space by half        

        Returns
        -------
        None.

        """

        super(InceptionTailRev, self).__init__()

        #build the initial upsampler
        self.up_sampler = nn.Sequential(
            nn.Conv1d(in_channels=num_feat, out_channels=num_feat*4,
                               kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(num_feat*4),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_feat*4, out_channels=num_feat*2,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_feat*2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_feat*2, out_channels=num_feat,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_feat, out_channels=num_feat,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU()
            )

        #divide hierarchically
        head_feat = int(num_feat/4)

        #three block
        self.three_layers = nn.ModuleList([ 
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=3, dilation=1, padding=1),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=3, dilation=2, padding=2),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=3, dilation=4, padding=4),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=3, dilation=8, padding=8)])
        self.three_bn = nn.BatchNorm1d(num_feat)

        #five block
        self.five_layers = nn.ModuleList([ 
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=5, dilation=1, padding=2),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=5, dilation=2, padding=4),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=5, dilation=4, padding=8),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=5, dilation=8, padding=16)])
        self.five_bn = nn.BatchNorm1d(num_feat)

        #seven block
        self.seven_layers = nn.ModuleList([ 
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=7, dilation=1, padding=3),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=7, dilation=2, padding=6),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=7, dilation=4, padding=12),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=7, dilation=8, padding=24)])
        self.seven_bn = nn.BatchNorm1d(num_feat)

        #nine block
        self.nine_layers = nn.ModuleList([ 
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=9, dilation=1, padding=4),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=9, dilation=2, padding=8),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=9, dilation=4, padding=16),
            nn.ConvTranspose1d(num_feat, head_feat, kernel_size=9, dilation=8, padding=32)])
        self.nine_bn = nn.BatchNorm1d(num_feat)

        #out block
        self.out_conv = nn.Sequential( 
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.Conv1d(num_feat, 1, kernel_size = 3, dilation=1, padding=1)
            )


    def forward(self,x):
        """
        forward method for the inception head module

        Parameters
        ----------
        x : Tensor
            input batch data of size nobs, 1, ntime.
            ntime typically 500

        Returns
        -------
        x : Tensor
            output following matrix multiplication
            87k parameters
            output batch data: nobs, 64 chan, 250 timepoints

        """

        #upsample to full scale
        x = self.up_sampler(x)
        
        #nine block
        x = torch.cat([layer(x) for layer in self.nine_layers], dim=1)
        x = self.nine_bn(x)
        x = F.relu(x)
        
        #seven block
        x = torch.cat([layer(x) for layer in self.seven_layers], dim=1)
        x = self.seven_bn(x)
        x = F.relu(x)
        
        #five block
        x = torch.cat([layer(x) for layer in self.five_layers], dim=1)
        x = self.five_bn(x)
        x = F.relu(x)

        #three block
        x = torch.cat([layer(x) for layer in self.three_layers], dim=1)
        x = self.three_bn(x)
        x = F.relu(x)

        #output block
        x = self.out_conv(x)
        x = torch.tanh(x.squeeze())
        return x
    

class MemnetMSE(nn.Module):
        def __init__(self, conv_in, conv_out, latent_dim):
            """
            Encoding residual block to take network to a managable size
            before implementing the transformers

            finally reduces the temporal space by 4 to about 50       

            Returns
            -------
            None.

            """
            
            super(MemnetMSE, self).__init__()
            
            self.conv_in = conv_in
            
            #Encoder Block
            self.conv_downsample = InceptionHead(conv_in, conv_out)
            self.encoder_downsample = nn.Sequential(
                EncoderDownSample(conv_out,conv_out*4),
                EncoderDownSample(conv_out,conv_out*16),
                EncoderDownSample(conv_out,conv_out*64))
            
            self.encoder_funnel = nn.Sequential(
                nn.Linear(7*conv_out, conv_out),
                nn.ReLU(),
                nn.Dropout()
                )
            self.fc_mu = nn.Linear(conv_out, latent_dim)
            self.fc_logvar = nn.Linear(conv_out, latent_dim)
            
            #diagnostic information
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 1)
                )
            
            #decoder block
            self.decoder_funnel = nn.Sequential(
                nn.Linear(latent_dim, 2112),
                nn.ReLU(),
                )
            self.decoder_upsample = nn.Sequential(
                EncoderUpSample(conv_in,conv_in*4),
                EncoderUpSample(conv_in,conv_in*16))
            
            self.conv_upsample = InceptionTail(conv_in)
            
            
        def encode(self, wave):
            
            x = self.conv_downsample(wave)
            #prepare for transformer encoder: batch, seq, feature
            x = self.encoder_downsample(x)
            x = x. reshape(x.shape[0],-1)
            #add the distribution to the linear layer
            x = self.encoder_funnel(x)
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            
            return mu, logvar
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def decode(self, z):
            x = self.decoder_funnel(z)
            x = x.reshape(x.shape[0],self.conv_in,-1)
            x = self.decoder_upsample(x)
            x = self.conv_upsample(x)
            return x
            
        def forward(self, batch):
            mu, logvar = self.encode(batch['wave'])
            #vary distribution
            z = self.reparameterize(mu, logvar)
            #reconstruct from perturbed latent space
            recon_x = self.decode(z)
            #compute from perturbed latent space
            class_logits = self.classifier(z).squeeze()
            
            #update dictionary with outputs
            batch_out = {
                'recon_x': recon_x,
                'mu': mu,
                'logvar': logvar,
                'class_logits': class_logits
                }
            batch.update(batch_out)

            return batch
        
        def loss_function(self, batch,
                          weight_lambda=1,
                          class_lambda=1,
                          recon_lambda=1,):
            
            #reconstruction loss - raw points
            RECON_WAVE = recon_lambda*F.mse_loss(batch['recon_x'],
                                    batch['wave'], reduction='sum')
            
            '''
            #reconstruction loss - frequency domain
            fft_recon_x = torch.fft.fft(batch['recon_x'])
            fft_x = torch.fft.fft(batch['wave'])
            
            #take up to 250Hz
            fft_recon_x = fft_recon_x[:, :125]
            fft_x = fft_x[:, :125]
            
            #compute the power
            power_recon_x = torch.abs(fft_recon_x) ** 2
            power_x =  torch.abs(fft_x) ** 2
            
            #normalize by largest value to preserve distirbution
            max_power_x , _ = torch.max(power_x, dim=1,
                                        keepdim=True)
            power_x = power_x / max_power_x
            max_power_recon_x , _ = torch.max(power_recon_x, dim=1,
                                              keepdim=True)
            power_recon_x = power_recon_x / max_power_recon_x
            
            #compute loss
            RECON_FFT = fft_lambda*F.mse_loss(power_recon_x,
                                              power_x, reduction='mean')
            '''
            
            #reparameterization loss
            KLD = -0.5 * torch.sum(1 + batch['logvar'] - 
                                   batch['mu'].pow(2) - 
                                   batch['logvar'].exp())
            #weight decay
            # L2 loss for all parameters
            l1_loss = sum(p.abs().sum() 
                          for p in self.encoder_downsample.parameters()) + \
                      sum(p.abs().sum() 
                          for p in self.decoder_upsample.parameters())
            l2_loss = l2_loss = sum(p.pow(2).sum() 
                                    for name, p in self.named_parameters()
                                    if 'conv_upsample.out_conv' not in name)
                  
            WEIGHT_LOSS = weight_lambda*(l1_loss+l2_loss)
            
            #reconstruction loss - distribution information
            '''
            CLASS_LOSS = class_lambda*F.binary_cross_entropy_with_logits(
                batch['class_logits'],
                batch['class_label'], 
                reduction='mean')
            
            #claculate classification accuracy
            predicted_class = (batch['class_logits'] > 0).long()
            correct_predictions = (predicted_class == batch['class_label'])
            num_correct = torch.sum(correct_predictions).cpu().item()
            CLASS_ACCURACY = (num_correct ,len(correct_predictions))
            
            #total loss
            TOTAL_LOSS = RECON_WAVE  + KLD + \
                WEIGHT_LOSS + CLASS_LOSS
            
            #put loss into a dictionary
            loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                          'RECON_WAVE': RECON_WAVE.cpu().item(),
                          'KLD': KLD.cpu().item(),
                          'WEIGHTS': WEIGHT_LOSS.cpu().item(),
                          'CLASS_LOSS': CLASS_LOSS.cpu().item(),
                          'CLASS_ACCURACY': CLASS_ACCURACY}
            
            '''
            TOTAL_LOSS = RECON_WAVE + KLD + WEIGHT_LOSS
            
            loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                          'RECON_WAVE': RECON_WAVE.cpu().item(),
                          'KLD': KLD.cpu().item(),
                          'WEIGHTS': WEIGHT_LOSS.cpu().item()}
            
            
            
            return TOTAL_LOSS, loss_dict
    
class MemnetFreq(MemnetMSE):
        def __init__(self, conv_in, conv_out, latent_dim):
            """
            Encoding residual block to take network to a managable size
            before implementing the transformers

            finally reduces the temporal space by 4 to about 50       

            Returns
            -------
            None.

            """
            
            super(MemnetFreq, self).__init__(conv_in, conv_out, latent_dim)
        
        def loss_function(self, batch,
                          weight_lambda=1,
                          class_lambda=1,
                          recon_lambda=1,
                          fft_lambda = 100):
            
            #reconstruction loss - raw points
            RECON_WAVE = recon_lambda*F.l1_loss(batch['recon_x'],
                                    batch['wave'], reduction='sum')
            
            #reconstruction loss - frequency domain
            fft_recon_x = torch.fft.fft(batch['recon_x'])
            fft_x = torch.fft.fft(batch['wave'])
            
            #take up to 250Hz
            fft_recon_x = fft_recon_x[:, :60]
            fft_x = fft_x[:, :60]
            
            #compute the power
            power_recon_x = torch.abs(fft_recon_x) ** 2
            power_x =  torch.abs(fft_x) ** 2
            
            #normalize by largest value to preserve distirbution
            max_power_x , _ = torch.max(power_x, dim=1,
                                        keepdim=True)
            power_x = power_x / max_power_x
            max_power_recon_x , _ = torch.max(power_recon_x, dim=1,
                                              keepdim=True)
            power_recon_x = power_recon_x / max_power_recon_x
            
            #compute loss
            RECON_FFT = fft_lambda*F.l1_loss(power_recon_x,
                                              power_x, reduction='sum')
            
            #reparameterization loss
            KLD = -0.5 * torch.sum(1 + batch['logvar'] - 
                                   batch['mu'].pow(2) - 
                                   batch['logvar'].exp())
            #weight decay
            # L2 loss for all parameters
            l1_loss = sum(p.abs().sum() 
                          for p in self.encoder_downsample.parameters()) + \
                      sum(p.abs().sum() 
                          for p in self.decoder_upsample.parameters())
            l2_loss = l2_loss = sum(p.pow(2).sum() 
                                    for name, p in self.named_parameters()
                                    if 'conv_upsample.out_conv' not in name)
                  
            WEIGHT_LOSS = weight_lambda*(l1_loss+l2_loss)
            
            #reconstruction loss - distribution information
            '''
            CLASS_LOSS = class_lambda*F.binary_cross_entropy_with_logits(
                batch['class_logits'],
                batch['class_label'], 
                reduction='mean')
            
            #claculate classification accuracy
            predicted_class = (batch['class_logits'] > 0).long()
            correct_predictions = (predicted_class == batch['class_label'])
            num_correct = torch.sum(correct_predictions).cpu().item()
            CLASS_ACCURACY = (num_correct ,len(correct_predictions))
            
            #total loss
            TOTAL_LOSS = RECON_WAVE  + KLD + \
                WEIGHT_LOSS + CLASS_LOSS
            
            #put loss into a dictionary
            loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                          'RECON_WAVE': RECON_WAVE.cpu().item(),
                          'KLD': KLD.cpu().item(),
                          'WEIGHTS': WEIGHT_LOSS.cpu().item(),
                          'CLASS_LOSS': CLASS_LOSS.cpu().item(),
                          'CLASS_ACCURACY': CLASS_ACCURACY}
            
            '''
            #TOTAL_LOSS = RECON_WAVE + RECON_FFT +  KLD + WEIGHT_LOSS
            TOTAL_LOSS = RECON_WAVE + RECON_FFT +  KLD 
            
            loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                          'RECON_WAVE': RECON_WAVE.cpu().item(),
                          'RECON_FFT': RECON_FFT.cpu().item(),
                          'KLD': KLD.cpu().item(),
                          'WEIGHTS': WEIGHT_LOSS.cpu().item()}
            
            return TOTAL_LOSS, loss_dict    


    
class InceptionTail(nn.Module):
    def __init__(self, num_feat):
        """
        Inception network inspired head for feature extraction
        uses residual connections to extract hierarcical temporal features

        finally reduces the temporal space by half        

        Returns
        -------
        None.

        """
        
        super(InceptionTail, self).__init__()
        
        #build the initial upsampler
        self.up_sampler = nn.Sequential(
            nn.Conv1d(in_channels=num_feat, out_channels=num_feat*4,
                               kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(num_feat*4),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_feat*4, out_channels=num_feat*2,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_feat*2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_feat*2, out_channels=num_feat,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_feat, out_channels=num_feat,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU()
            )
        
        #divide hierarchically
        head_feat = int(num_feat/4)
        
        #three block
        self.three_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=3, dilation=8, padding=8)])
        self.three_bn = nn.BatchNorm1d(num_feat)
        
        #five block
        self.five_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=1, padding=2),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=2, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=4, padding=8),
            nn.Conv1d(num_feat, head_feat, kernel_size=5, dilation=8, padding=16)])
        self.five_bn = nn.BatchNorm1d(num_feat)
        
        #seven block
        self.seven_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=1, padding=3),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=2, padding=6),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=4, padding=12),
            nn.Conv1d(num_feat, head_feat, kernel_size=7, dilation=8, padding=24)])
        self.seven_bn = nn.BatchNorm1d(num_feat)
        
        #nine block
        self.nine_layers = nn.ModuleList([ 
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=1, padding=4),
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=2, padding=8),
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(num_feat, head_feat, kernel_size=9, dilation=8, padding=32)])
        self.nine_bn = nn.BatchNorm1d(num_feat)
        
        #out block
        self.out_conv = nn.Sequential( 
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(),
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(), 
            nn.Conv1d(num_feat, num_feat, kernel_size = 3, dilation=1, padding=1),
            nn.BatchNorm1d(num_feat),
            nn.ReLU(), 
            nn.Conv1d(num_feat, 1, kernel_size = 3, dilation=1, padding=1))

        
    def forward(self,x):
        """
        forward method for the inception head module

        Parameters
        ----------
        x : Tensor
            input batch data of size nobs, 1, ntime.
            ntime typically 500

        Returns
        -------
        x : Tensor
            output following matrix multiplication
            87k parameters
            output batch data: nobs, 64 chan, 250 timepoints

        """
        
        #upsample to full scale
        x = self.up_sampler(x)
        
        #maintain gradient
        resid = x
        
        #three block
        x = torch.cat([layer(x) for layer in self.three_layers], dim=1)
        x = self.three_bn(x)
        x = F.relu(x)
        
        #five block
        x = torch.cat([layer(x) for layer in self.five_layers], dim=1)
        x = self.five_bn(x)
        x = F.relu(x + resid)
        resid = x
        
        #seven block
        x = torch.cat([layer(x) for layer in self.seven_layers], dim=1)
        x = self.seven_bn(x)
        x = F.relu(x)
        
        #nine block
        x = torch.cat([layer(x) for layer in self.nine_layers], dim=1)
        x = self.nine_bn(x)
        x = F.relu(x + resid)

        #output block
        x = self.out_conv(x)
        x = torch.tanh(x.squeeze())
        return x
    