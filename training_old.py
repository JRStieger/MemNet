# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:00:26 2024

training module for MemNet package

@author: jstieger
"""
from collections import defaultdict
import copy
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import time
import torch
from torch.utils.data import Sampler
from tqdm import tqdm


def train_memnet_batch(batch, model, optimizer, device, epoch,
                       clip_value = 1.0):
    #process batched data
    batch_reshaped = [ tensor.view(-1,tensor.shape[-1]) if tensor.dim() == 3 
                   else tensor.view(-1) for tensor in batch]
    wave_source, wave_target, dist, loc, class_label = batch_reshaped
    
    #send to device
    wave_source = wave_source.float().to(device)
    wave_target = wave_target.float().to(device)
    dist = dist.float().to(device)
    loc = loc.float().to(device)
    class_label = class_label.long().to(device)
    
    #package in dict
    batch_in = {'wave_source': wave_source,
                'wave_target': wave_target,
                'dist': dist,
                'loc': loc,
                'class_label': class_label
                }
    
    #prepare training
    model.train()
    optimizer.zero_grad()
    #forward pass
    batch_out = model(batch_in)
    loss, loss_dict = model.loss_function(batch_out)
    #send loss backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


@torch.no_grad()
def validate_memnet_batch(batch, model, device, epoch):
    #process batched data
    batch_reshaped = [ tensor.view(-1,tensor.shape[-1]) if tensor.dim() == 3 
                   else tensor.view(-1) for tensor in batch]
    wave_source, wave_target, dist, loc, class_label = batch_reshaped
    
    #send to device
    wave_source = wave_source.float().to(device)
    wave_target = wave_target.float().to(device)
    dist = dist.float().to(device)
    loc = loc.float().to(device)
    class_label = class_label.long().to(device)
    
    #package in dict
    batch_in = {'wave_source': wave_source,
                'wave_target': wave_target,
                'dist': dist,
                'loc': loc,
                'class_label': class_label
                }
    
    #prepare training
    model.eval()
    #forward pass
    batch_out = model(batch_in)
    loss, loss_dict = model.loss_function(batch_out)
    
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


def train_AE_batch(batch, model, optimizer, device, epoch,
                       clip_value = 1.0, batch_size = 2048):
    
    #prepare training
    model.train()
    
    #empty batch dict
    batch_loss_dict = []
    
    #unpack batch
    wave_source, wave_target, dist, loc, class_label = batch
    
    #get random index
    num_samples = wave_source.shape[0]
    shuffled_indices = torch.randperm(num_samples)
    batches = [shuffled_indices[i:i + batch_size] 
                 for i in range(0, num_samples, batch_size)]
    
    #process mini-batches
    for batch_ind in batches:
    
        #send to device
        wave_source_b = wave_source[batch_ind].float().to(device)
        wave_target_b = wave_target[batch_ind].float().to(device)

        #package in dict
        batch_in = {'wave_source': wave_source_b,
                    'wave_target': wave_target_b
                    }
    
        optimizer.zero_grad()
        #forward pass
        batch_out = model(batch_in)
        loss, loss_dict = model.loss_function(batch_out)
        if not batch_loss_dict:
            batch_loss_dict = {key: [] for key in loss_dict.keys()}
            
        for key in loss_dict.keys():
            batch_loss_dict[key].append(loss_dict[key])
        #send loss backwards
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
    #return parameters - average loss
    for key in batch_loss_dict.keys():
        batch_loss_dict[key] = np.mean(batch_loss_dict[key])
    
    loss_dict['epoch'] = epoch
    return loss_dict


@torch.no_grad()
def validate_AE_batch(batch, model, device, epoch, batch_size = 2048):

    #prepare training
    model.eval()    

    #empty batch dict
    batch_loss_dict = []
    
    #unpack batch
    wave_source, wave_target, dist, loc, class_label = batch
    
    #get random index
    num_samples = wave_source.shape[0]
    shuffled_indices = torch.randperm(num_samples)
    batches = [shuffled_indices[i:i + batch_size] 
                 for i in range(0, num_samples, batch_size)]
    
    #process mini-batches
    for batch_ind in batches:
    
        #send to device
        wave_source_b = wave_source[batch_ind].float().to(device)
        wave_target_b = wave_target[batch_ind].float().to(device)

        #package in dict
        batch_in = {'wave_source': wave_source_b,
                    'wave_target': wave_target_b
                    }
        
        #forward pass
        batch_out = model(batch_in)
        loss, loss_dict = model.loss_function(batch_out)
        if not batch_loss_dict:
            batch_loss_dict = {key: [] for key in loss_dict.keys()}
            
        for key in loss_dict.keys():
            batch_loss_dict[key].append(loss_dict[key])
        
    #return parameters - average loss
    for key in batch_loss_dict.keys():
        batch_loss_dict[key] = np.mean(batch_loss_dict[key])
    
    loss_dict['epoch'] = epoch
    return loss_dict


class StratifiedBatchSampler(Sampler):
    def __init__(self, data_table, batch_size, training = True):
        self._data_table = data_table[
            data_table.training == training].reset_index(drop=True)
        self.batch_size = batch_size

    def __iter__(self):
        
        
        class_labels = self._data_table['class_label']
        label_to_indices = defaultdict(list)
        
        # Group indices by class
        for index, label in enumerate(class_labels):
            label_to_indices[label].append(index)
        
        # Shuffle indices within each class
        for label in label_to_indices:
            np.random.shuffle(label_to_indices[label])
        
        # Calculate number of batches
        num_batches = math.ceil(len(class_labels) / self.batch_size)
        
        # Create empty batches
        batches = [[] for _ in range(num_batches)]
        
        # Distribute indices to batches while maintaining class distribution
        for label, indices in label_to_indices.items():
            for i, index in enumerate(indices):
                batches[i % num_batches].append(index)
                
        # Shuffle the order of indices within each batch
        for batch in batches:
            np.random.shuffle(batch)
            
        #stack them together
        idx_list = np.concatenate(batches)
        yield from idx_list
        
    def __len__(self):
        
        return len(self._data_table)


class VaeHistory:
    def __init__(self, keys):
        
        self._keys = keys
        self._epoch_dict = {}
        self._batch_dict = {}
        for d_type in ('train','val'):
            self._epoch_dict[d_type] = {key: [] for key in keys}
            self._batch_dict[d_type] = {key: [] for key in keys}
            
            
    def reset(self):
        
        for d_type in ('train','val'):
            self._batch_dict[d_type] = {key: [] for key in self._keys}
            
    def digest(self):
        
        for d_type in ('train','val'):
            for key in self._keys:
                if key == 'CLASS_ACCURACY':
                    num_correct = sum(t[0] for t in 
                                      self._batch_dict[d_type][key])
                    tot_obs = sum(t[1] for t in 
                                      self._batch_dict[d_type][key])
                    value = num_correct/tot_obs
                else:
                    value = np.mean(self._batch_dict[d_type][key])
                    
                self._epoch_dict[d_type][key].append(value)
        
        self.print_epoch()
        self.reset()
        
        
    def update(self, d_type, batch_dict):
        
        for key in batch_dict.keys():
            self._batch_dict[d_type][key].append(batch_dict[key])
            
    def get_dict(self, d_type):
        
        return self._epoch_dict[d_type]
    
    def validation_value(self, key):
        
        return self._epoch_dict['val'][key][-1]
    
    def training_value(self, key):
        
        return self._epoch_dict['train'][key][-1]
    
    def plot_history(self):
        """
        function finds all the diagnostic keys and prints each in a separate 
        matplotlib line plot

        Returns
        -------
        None.

        """
        
        #pull out dictionaries
        train_dict = self.get_dict('train')
        val_dict = self.get_dict('val')
        
        #zip the two for plotting
        epochs = train_dict['epoch']
        metrics_dict = {key: (epochs, train_dict[key], val_dict[key])
                        for key in train_dict.keys()
                        if key != 'epoch'}
        
        for key, metrics in metrics_dict.items():
            self._plot_training_metrics(key,metrics)
    
    def _plot_training_metrics(self, key, metrics):
        """
        function plots the metrics from the training history

        Parameters
        ----------
        key : str
            name of the metric.
        metrics : tuple
            contains (epochs, training data, validation data).
            all lists

        Returns
        -------
        None.

        """
        
        # Set font properties
        prop = self.prop

        # Unpack metrics
        epochs, train_values, val_values = metrics

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Plot data on the first subplot
        ax1.plot(epochs, train_values, label='Train', color='blue', linewidth=1.5)
        ax1.plot(epochs, val_values, label='Validation', color='red', linewidth=1.5)
        
        # Title and labels for the first subplot
        ax1.set_title(f'{key} Over Epochs (Linear Scale)', fontsize=16, loc='left')
        ax1.set_xlabel('Epochs', fontsize=14, fontproperties=prop)
        ax1.set_ylabel(key, fontsize=14, fontproperties=prop)
        
        # Legend for the first subplot
        ax1.legend(fontsize=18, prop=prop, loc='upper right', ncol=2, bbox_to_anchor=(1, 1))
        
        # Plot data on the second subplot with semilogy
        ax2.semilogy(epochs, train_values, label='Train', color='blue', linewidth=1.5)
        ax2.semilogy(epochs, val_values, label='Validation', color='red', linewidth=1.5)
        
        # Title and labels for the second subplot
        ax2.set_title(f'{key} Over Epochs (Semilog Scale)', fontsize=16, loc='left')
        ax2.set_xlabel('Epochs', fontsize=14, fontproperties=prop)
        ax2.set_ylabel(key, fontsize=14, fontproperties=prop)
        
        # Legend for the second subplot
        ax2.legend(fontsize=18, prop=prop, loc='upper right', ncol=2, bbox_to_anchor=(1, 1))
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
        
        
    def print_epoch(self):
        print()
        print('='*100)
        for d_type in ('train','val'):
            if d_type == 'val': 
                print('-'*100)
            print(f'{d_type} data:')
            for key in self._keys:
                value = self._epoch_dict[d_type][key][-1]
                if key == 'epoch':
                    print(f'{key}: {value}')
                else:
                    print(f'{key}: {value:.4f}')
                
        print('='*100)
        print()
        
    def print_batch(self, batch):
        print()
        print('='*100)
        d_type = 'train'
        print(f'{d_type} data batch {batch}:')
        for key in self._keys:
            if key == 'CLASS_ACCURACY':
                num_correct = sum(t[0] for t in 
                                  self._batch_dict[d_type][key])
                tot_obs = sum(t[1] for t in 
                                  self._batch_dict[d_type][key])
                value = num_correct/tot_obs
            else:
                value = np.mean(self._batch_dict[d_type][key])
            print(f'{key}: {value:.4f}')
                
        print('='*100)
        print()
        
        
        
class TrainingDiagnostics:
    def __init__(self, model, history, val_dl,
                 vae = True, nbatches = 1000):
        """
        Class provides methods to evaluate model training

        Parameters
        ----------
        model : Pytorch model class
            model to examine
            provides these methods:
                weight_hist: looks at histograms of weights
                latent_hist: examines latent space of zero mean unit variance
                class_hist: plots histograms of latent space for each class
        history : VAEHistory
            history class to plot training metrics of training and validation.
            provides these methods:
                plot_history: plots training and validation for all metrics 
                    in history class

        Returns
        -------
        None.

        """
        
        self._model = model
        self.history = history
        
        self.vae = vae
        
        #containers for validation diagnostic data
        self._val_features = ['orig_data',
                              'latent_data',
                              'recon_data',
                              'class_label']
        self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._data_dict = {key: [] for key in self._val_features}
        self._decompose_validation_data(val_dl,nbatches)
        
        #extract parameter weights
        self._param_weights = self._extract_weights()
        
        #font properties
        font_path = r"C:\Windows\Fonts\Brink - BR Cobane Regular.otf"
        self.prop = fm.FontProperties(fname=font_path)
        
    def _decompose_validation_data(self, val_dl, nbatches):
        """
        function computes the latent space and reconstructions for nbatches
        in the validation dataloader val_dl

        Parameters
        ----------
        val_dl : dataloader
            data loader for the validation set.
        nbatches : int
            maximum number of batches to include 
            in the internal data structures.

        Returns
        -------
        None.

        """
        
        print("Processing validation data...")
        #loop through data
        for batch_idx, batch in tqdm(enumerate(val_dl)):
            batch_dict = self._memnet_recon_batch(batch,
                                                  self._model,
                                                  self.device)
            #update data dictionary
            for key in batch_dict.keys():
                self._data_dict[key].append(batch_dict[key])
            
            #set maximum batches
            if batch_idx > nbatches:
                break
            
        #stack into one numpy array
        for key in self._data_dict.keys():
            self._data_dict[key] = np.concatenate(
                self._data_dict[key], axis = 0)
            
    def _extract_weights(self):
        """
        function extracts the weights from the model
        used to extract histograms of the parameter values

        Returns
        -------
        weights : dictionary holding all of the weight parameters

        """
        weights = {}
        for name, module in self._model.named_modules():
            if isinstance(module, (torch.nn.TransformerEncoderLayer,
                                   torch.nn.Conv1d,
                                   torch.nn.ConvTranspose1d,
                                   torch.nn.Linear)):
                if hasattr(module, 'weight'):
                    weights[f'{name}.weight'] = \
                        module.weight.data.cpu().numpy()
        return weights
        
            
    @torch.no_grad()
    def _memnet_recon_batch(self, batch, model, device):
        """
        function processes the validation data from the model for plotting
        processes one batch at a time

        Parameters
        ----------
        batch : tuple
            batch data.
        model : pytorch model
        device : device
            where to send the data.

        Returns
        -------
        batch_dict : data output from one batch

        """
        #process batched data
        batch_reshaped = [ tensor.view(-1,tensor.shape[-1]) if tensor.dim() == 3 
                       else tensor.view(-1) for tensor in batch]
        wave_source, wave_target, dist, loc, class_label = batch_reshaped
        wave = wave_target.float().to(device)
        
        #prepare training
        model.eval()
        #forward pass
        if self.vae:
            latent_data, logvar = model.encode(wave)
        else:
            latent_data = model.encode(wave)
        recon_data = model.decode(latent_data)
        
        batch_dict = {
            'orig_data': wave.cpu().numpy(),
            'latent_data': latent_data.cpu().numpy(),
            'recon_data': recon_data.cpu().numpy(),
            'class_label': class_label.cpu().numpy()}

        return batch_dict
        
    def plot_history(self):
        """
        function finds all the diagnostic keys and prints each in a separate 
        matplotlib line plot

        Returns
        -------
        None.

        """
        
        #call history method
        self.history.plot_history('train')
        
            
    def plot_latent_data(self, combined_plot = True, by_class = False):
        """
        function plots histograms of the latent data to examine for
        1. zero mean, unit variance
        2. class separability in different dimensions

        Parameters
        ----------
        combined_plot : Bool, optional
            plot all dimensions on one plot; easy to see them all
            The default is True.
        by_class : Bool, optional
            plot data as different colored histograms to examine
            class separability
            The default is False.

        Returns
        -------
        None.

        """
        latent_data = self._data_dict['latent_data']
        n_dims = latent_data.shape[1]
        
        # Set font properties
        prop = self.prop

        if by_class:
            class_label = self._data_dict['class_label']
            
            # Unique class labels
            unique_labels = np.unique(class_label)
            colors = plt.get_cmap('Set1')(
                range(len(unique_labels)))
            
            if combined_plot:
                # Create two figures with 25 subplots each
                fig1, axs1 = plt.subplots(5, 5, figsize=(15, 8))
                fig2, axs2 = plt.subplots(5, 5, figsize=(15, 8))
                
                # Flatten the axes array for easy iteration
                axs1 = axs1.flatten()
                axs2 = axs2.flatten()
                
                # Plot histograms for the first 25 dimensions in the first figure
                for i in range(25):
                    for j, label in enumerate(unique_labels):
                        axs1[i].hist(latent_data[class_label == label, i],
                                     bins=500, alpha=0.5, color=colors[j],
                                     label=f'Class {label}')
                    axs1[i].set_title(f'Dimension {i+1}', loc='left',
                                      fontproperties=prop)
                    axs1[i].legend(fontsize='x-small')
                
                # Plot histograms for the next 25 dimensions in the second figure
                for i in range(25, min(50, n_dims)):
                    for j, label in enumerate(unique_labels):
                        axs2[i-25].hist(latent_data[class_label == label, i], 
                                        bins=500, alpha=0.5, color=colors[j],
                                        label=f'Class {label}')
                    axs2[i-25].set_title(f'Dimension {i+1}', loc='left',
                                         fontproperties=prop)
                    axs2[i-25].legend(fontsize='x-small')
                
                # Adjust layout to prevent overlap
                fig1.tight_layout()
                fig2.tight_layout()
                
                # Show the plots
                plt.show()
            else:
                # Create a figure for each column in latent_data
                for i in range(n_dims):
                    fig, ax = plt.subplots(figsize=(15, 8))
                    
                    for j, label in enumerate(unique_labels):
                        ax.hist(latent_data[class_label == label, i], bins=500, 
                                alpha=0.5, color=colors[j],
                                label=f'Class {label}')
                    
                    ax.set_title(f'Dimension {i+1}', loc='left',
                                 fontproperties=prop)
                    ax.legend(fontsize='x-small')
                    
                    # Adjust layout to prevent overlap
                    fig.tight_layout()
                    
                    # Show the plot
                    plt.show()
            
        else: #just latent space
            if combined_plot:
                # Create two figures with 25 subplots each
                fig1, axs1 = plt.subplots(5, 5, figsize=(15, 8))
                fig2, axs2 = plt.subplots(5, 5, figsize=(15, 8))
                
                # Flatten the axes array for easy iteration
                axs1 = axs1.flatten()
                axs2 = axs2.flatten()
                
                # Plot histograms for the first 25 dimensions in the first figure
                for i in range(25):
                    axs1[i].hist(latent_data[:, i], bins=500)
                    axs1[i].set_title(f'Dimension {i+1}', loc='left',
                                      fontproperties=prop)
                
                # Plot histograms for the next 25 dimensions in the second figure
                for i in range(25, min(50, n_dims)):
                    axs2[i-25].hist(latent_data[:, i], bins=500)
                    axs2[i-25].set_title(f'Dimension {i+1}', loc='left',
                                         fontproperties=prop)
                
                # Adjust layout to prevent overlap
                fig1.tight_layout()
                fig2.tight_layout()
                
                # Show the plots
                plt.show()
                
            else:
                for i in range(n_dims):
                    fig, ax = plt.subplots(figsize=(15, 8))
                    ax.hist(latent_data[:, i], bins=500)
                    ax.set_title(f'Dimension {i+1}', loc='left',
                                  fontproperties=prop)
        
    def plot_recon_data(self, nplots):
        
        #pull out data
        orig_data = self._data_dict['orig_data']
        recon_data = self._data_dict['recon_data']
        prop = self.prop
        
        #shuffle
        n_obs = orig_data.shape[0]
        indices = np.random.permutation(n_obs)
        
        # Set up the loop to plot each pair of rows
        for i in range(nplots):
            plot_i = indices[i]
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot original data as blue and reconstructed data as red
            ax.plot(orig_data[plot_i], color='blue', label='Original')
            ax.plot(recon_data[plot_i], color='red', label='Reconstructed')
            
            # Set the legend with the specified font properties
            ax.legend(prop=prop)
            
            # Draw a thin transparent grey line at zero across the figure
            ax.axhline(0, color='grey', linewidth=0.5, alpha=0.5)
            
            # Show the plot
            plt.show()
            
    def plot_weights(self):
        """
        function plots histograms of the weight values in the network

        Returns
        -------
        None.

        """
        
        prop = self.prop
        for name, weight in self._param_weights.items():
            
            #open new figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            #flatten weights
            
            weight_flat = weight.flatten()
            if len(weight_flat) > 1000:
                nbins = 500
            else:
                nbins = 50
                
            ax.hist(weight_flat, bins=nbins)
            ax.set_title(f'Weights for {name}', fontproperties=prop, 
                         fontsize=18, loc='left')
            plt.xlabel('Weight Value', fontproperties=prop, fontsize=12)
            plt.ylabel('Frequency', fontproperties=prop, fontsize=12)
            plt.show()
            
            
        



    