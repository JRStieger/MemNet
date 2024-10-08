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
from memnet.datasets import AllSbjData
from memnet.datasets import get_sbj_table_pretrain, get_sbj_table_JEPA
from memnet.masks import apply_masks, MaskCollatorWave
from memnet.models import copy_state_dict, PretrainTransformer, ProbabilisticTransformer
from memnet.models import JEPAEncoder, JEPAPredictor
from memnet.scheduler import WarmupCosineSchedule, CosineWDSchedule
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import random
from scipy.special import softmax
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d



def init_JEPA_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr=0.0001,
    ref_lr=0.001,
    warmup=15,
    num_epochs=500,
    wd=0.04,
    final_wd=0.4,
    final_lr=1e-6,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    
    return optimizer, scheduler, wd_scheduler

def init_pretrain_opt(
    model,
    iterations_per_epoch,
    start_lr=0.0001,
    ref_lr=0.001,
    warmup=15,
    num_epochs=500,
    wd=0.04,
    final_wd=0.4,
    final_lr=1e-6,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in model.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)
                       and p.requires_grad)
        }, {
            'params': (p for n, p in model.named_parameters()
                       if (('bias' in n) or (len(p.shape) == 1)) and
                       (p.requires_grad)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    
    return optimizer, scheduler, wd_scheduler


def unfreeze_model(model, train_dl):
    '''
    function takes a frozen model then unfreezes the parameters
    and re-initializes the optimizers

    Parameters
    ----------
    model : transformer pretrain model
    train_dl : training data loader to reinitialize optimizers

    Returns
    -------
    model : returns the model with unfrozen layers
    optimizer : new optimizer with access to all parameters
    scheduler : new scheduler
    wd_scheduler : new weight decay scheduler

    '''
    # Unfreeze the specified layers in WavePredictor
    for param in model.patch_embed.parameters():
        param.requires_grad = True
    
    model.pos_embed.requires_grad = True
    
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = True
            
    #reinitialize optimizer with full set of parameters
    nepochs = 300
    warmup = 15
    optimizer, scheduler, wd_scheduler = init_pretrain_opt(
        model,
        len(train_dl),
        start_lr=0.0001,
        ref_lr=0.001,
        warmup=warmup,
        num_epochs=nepochs,
        wd=0.04,
        final_wd=0.4,
        final_lr=1e-6,
        )
    
    #step to commesurate LR
    for e in range(101):
        for batch_idx in range(len(train_dl)):
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
            
    return model, optimizer, scheduler, wd_scheduler

def train_IC_batch(batch, model, optimizer, device, epoch,
                   criterion):
    
    #pull out batch data
    wave, class_labels = batch
    
    #send to device
    wave = wave.float().to(device)
    class_labels = class_labels.float().to(device)

    #prepare training
    model.train()
    optimizer.zero_grad()
    #forward pass
    logits = model(wave)
    
    #get weighted loss
    '''
    per_sample_loss = criterion(logits, class_labels)
    batch_weights = class_weights[class_labels.long()]
    weighted_loss = per_sample_loss * batch_weights
    loss = weighted_loss.mean()
    '''
    loss = criterion(logits, class_labels)
    
    # Calculate L1 and L2 regularization
    '''
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    
    # Add L1 and L2 regularization to the loss
    loss = loss + 0.00001 * l1_norm + 0.0001 * l2_norm
    '''
    
    #send loss backwards
    loss.backward()
    optimizer.step()
    
    # make batch targets/preds
    target_pred = (class_labels.long().cpu().numpy(),
                   logits.detach().cpu().numpy())
    
    #return dictionary
    loss_dict = {'epoch': epoch,
                 'CLASS_LOSS': loss.cpu().item(),
                 'ROC_AUC': target_pred,
                 'F1_SCORE': target_pred}
                 
    return loss_dict


@torch.no_grad()
def validate_IC_batch(batch, model, device, epoch,
                   criterion):
    
    #pull out batch data
    wave, class_labels = batch
    
    #send to device
    wave = wave.float().to(device)
    class_labels = class_labels.float().to(device)

    #prepare training
    model.eval()

    #forward pass
    logits = model(wave)
    #get weighted loss
    '''
    per_sample_loss = criterion(logits, class_labels)
    batch_weights = class_weights[class_labels.long()]
    weighted_loss = per_sample_loss * batch_weights
    loss = weighted_loss.mean()
    '''
    
    loss = criterion(logits, class_labels)
    
    '''
    # Calculate L1 and L2 regularization
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    
    # Add L1 and L2 regularization to the loss
    loss = loss + 0.00001 * l1_norm + 0.0001 * l2_norm
    '''
    # make batch targets/preds
    target_pred = (class_labels.long().cpu().numpy(),
                   logits.detach().cpu().numpy())
    
    #return dictionary
    loss_dict = {'epoch': epoch,
                 'CLASS_LOSS': loss.cpu().item(),
                 'ROC_AUC': target_pred,
                 'F1_SCORE': target_pred}
    
    return loss_dict

@torch.no_grad()
def feat_from_jepa(batch, jepa_model, device):
    
    #pull out batch data
    wave, class_labels = batch
    
    #send to device
    wave = wave.float().to(device)

    #prepare training
    jepa_model.eval()

    #forward pass
    features = jepa_model(wave)
    
    return [features, class_labels]

# KL Divergence Loss Function
def kl_divergence(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


def train_pred_batch(batch, model, optimizer, device, epoch, class_weights):
    
    #pull out batch data
    wave, class_labels = batch
    
    #send to device
    wave = wave.float().to(device)
    class_labels = class_labels.long().to(device)

    #prepare training
    model.train()
    optimizer.zero_grad()
    #forward pass
    logits = model(wave)
    loss, loss_dict = classification_loss(logits, class_labels, class_weights)
    #send loss backwards
    loss.backward()
    optimizer.step()
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


@torch.no_grad()
def validate_pred_batch(batch, model, device, epoch, class_weights):
    
    #pull out batch data
    wave, class_labels = batch
    
    #send to device
    wave = wave.float().to(device)
    class_labels = class_labels.long().to(device)

    #prepare training
    model.eval()
    #forward pass
    logits = model(wave)
    loss, loss_dict = classification_loss(logits, class_labels, class_weights)
    
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


def classification_loss(logits, class_labels, class_weights):

    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        weight=class_weights,
        reduction='mean')
    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        reduction='mean')
    
    #claculate classification accuracy
    predicted_class = torch.argmax(
        logits, dim = -1).long()
    correct_predictions = (predicted_class == class_labels)
    num_correct = torch.sum(correct_predictions).cpu().item()
    CLASS_ACCURACY = (num_correct ,len(correct_predictions))
    
    #scikit learn prep
    target_pred = (class_labels.long().cpu().numpy(),
                   logits.detach().cpu().numpy())

    loss_dict = {'CLASS_LOSS': CLASS_LOSS.cpu().item(),
                  'CLASS_ACCURACY': CLASS_ACCURACY,
                  'ROC_AUC': target_pred,
                  'F1_SCORE': target_pred}
    
    return CLASS_LOSS, loss_dict


def train_pred_batch_weight(batch, model, optimizer, device, epoch):
    
    #pull out batch data
    wave, class_labels, weights = batch
    
    #send to device
    wave = wave.float().to(device)
    weights = weights.float().to(device)
    class_labels = class_labels.long().to(device)

    #prepare training
    model.train()
    optimizer.zero_grad()
    #forward pass
    logits = model(wave)
    loss, loss_dict = classification_loss_weight(logits, class_labels, weights)
    #send loss backwards
    loss.backward()
    optimizer.step()
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


@torch.no_grad()
def validate_pred_batch_weight(batch, model, device, epoch):
    
    #pull out batch data
    wave, class_labels, weights = batch
    
    #send to device
    wave = wave.float().to(device)
    weights = weights.float().to(device)
    class_labels = class_labels.long().to(device)

    #prepare training
    model.eval()
    #forward pass
    logits = model(wave)
    loss, loss_dict = classification_loss_weight(logits, class_labels, weights)
    
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


def classification_loss_weight(logits, class_labels, weights):

    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        weight=class_weights,
        reduction='mean')
    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        reduction='none')
    CLASS_LOSS = torch.mean(CLASS_LOSS*weights)
    
    #claculate classification accuracy
    predicted_class = torch.argmax(
        logits, dim = -1).long()
    correct_predictions = (predicted_class == class_labels)
    num_correct = torch.sum(correct_predictions).cpu().item()
    CLASS_ACCURACY = (num_correct ,len(correct_predictions))
    
    #scikit learn prep
    target_pred = (class_labels.long().cpu().numpy(),
                   logits.detach().cpu().numpy())

    loss_dict = {'CLASS_LOSS': CLASS_LOSS.cpu().item(),
                  'CLASS_ACCURACY': CLASS_ACCURACY,
                  'ROC_AUC': target_pred,
                  'F1_SCORE': target_pred}
    
    return CLASS_LOSS, loss_dict


def train_prob_batch(batch, model, optimizer, device, epoch, class_weights, clip_value=1.0):
    
    #pull out batch data
    wave, class_labels = batch
    
    #send to device
    wave = wave.float().to(device)
    class_labels = class_labels.long().to(device)

    #prepare training
    model.train()
    optimizer.zero_grad()
    #forward pass
    logits, mean, logvar = model(wave)
    loss, loss_dict = classification_loss_prob(logits,
                                               class_labels,
                                               mean,
                                               logvar)
    #send loss backwards
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


@torch.no_grad()
def validate_prob_batch(batch, model, device, epoch, class_weights):
    
    #pull out batch data
    wave, class_labels = batch
    
    #send to device
    wave = wave.float().to(device)
    class_labels = class_labels.long().to(device)

    #prepare training
    model.eval()
    #forward pass
    logits, mean, logvar = model(wave)
    loss, loss_dict = classification_loss_prob(logits,
                                               class_labels,
                                               mean,
                                               logvar
                                               )
    
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


def classification_loss_prob(logits, class_labels, mean, logvar):

    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        weight=class_weights,
        reduction='mean')
    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        reduction='mean')
    
    KL_LOSS = kl_divergence(mean, logvar)
    
    
    TOTAL_LOSS = CLASS_LOSS + 0.01*KL_LOSS
    
    #claculate classification accuracy
    predicted_class = torch.argmax(
        logits, dim = -1).long()
    correct_predictions = (predicted_class == class_labels)
    num_correct = torch.sum(correct_predictions).cpu().item()
    CLASS_ACCURACY = (num_correct ,len(correct_predictions))
    
    #scikit learn prep
    target_pred = (class_labels.long().cpu().numpy(),
                   logits.detach().cpu().numpy())

    loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                'CLASS_LOSS': CLASS_LOSS.cpu().item(),
                'KL_LOSS': KL_LOSS.cpu().item(),
                  'CLASS_ACCURACY': CLASS_ACCURACY,
                  'ROC_AUC': target_pred,
                  'F1_SCORE': target_pred}
    
    return TOTAL_LOSS, loss_dict


def train_prob_batch_weight(batch, model, optimizer, device, epoch):
    
    #pull out batch data
    wave, class_labels, weights = batch
    
    #send to device
    wave = wave.float().to(device)
    weights = weights.float().to(device)
    class_labels = class_labels.long().to(device)

    #prepare training
    model.train()
    optimizer.zero_grad()
    #forward pass
    logits, mean, logvar = model(wave)
    loss, loss_dict = classification_loss_prob_weight(logits,
                                               class_labels,
                                               mean,
                                               logvar,
                                               weights)
    #send loss backwards
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


@torch.no_grad()
def validate_prob_batch_weight(batch, model, device, epoch):
    
    #pull out batch data
    wave, class_labels, weights = batch
    
    #send to device
    wave = wave.float().to(device)
    weights = weights.float().to(device)
    class_labels = class_labels.long().to(device)
    
    #prepare training
    model.eval()
    #forward pass
    logits, mean, logvar = model(wave)
    loss, loss_dict = classification_loss_prob_weight(logits,
                                               class_labels,
                                               mean,
                                               logvar,
                                               weights)
    
    #return parameters
    loss_dict['epoch'] = epoch
    return loss_dict


def classification_loss_prob_weight(logits, class_labels, mean, logvar, weights):

    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        weight=class_weights,
        reduction='mean')
    '''
    #calculate the classification loss for valid
    CLASS_LOSS = F.cross_entropy(
        logits,
        class_labels,
        reduction='none')
    CLASS_LOSS = torch.mean(CLASS_LOSS*weights)
    
    KL_LOSS = kl_divergence(mean, logvar)
    
    
    TOTAL_LOSS = CLASS_LOSS + 0.01*KL_LOSS
    
    #claculate classification accuracy
    predicted_class = torch.argmax(
        logits, dim = -1).long()
    correct_predictions = (predicted_class == class_labels)
    num_correct = torch.sum(correct_predictions).cpu().item()
    CLASS_ACCURACY = (num_correct ,len(correct_predictions))
    
    #scikit learn prep
    target_pred = (class_labels.long().cpu().numpy(),
                   logits.detach().cpu().numpy())

    loss_dict = {'TOTAL_LOSS': TOTAL_LOSS.cpu().item(),
                'CLASS_LOSS': CLASS_LOSS.cpu().item(),
                'KL_LOSS': KL_LOSS.cpu().item(),
                  'CLASS_ACCURACY': CLASS_ACCURACY,
                  'ROC_AUC': target_pred,
                  'F1_SCORE': target_pred}
    
    return TOTAL_LOSS, loss_dict

def forward_target(target_encoder, wave, masks_pred):
    with torch.no_grad():
        h = target_encoder(wave)
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
        # -- create targets (masked regions of h)
        h = apply_masks(h, masks_pred)
        return h
    
    
def forward_context(encoder, predictor, wave, masks_enc, masks_pred):
    z = encoder(wave, masks_enc)
    z = predictor(z, masks_enc, masks_pred)
    return z


def loss_fn(z, h):
    loss = F.smooth_l1_loss(z, h)
    loss_dict = {'TOTAL_LOSS': loss.cpu().item()}
    return loss, loss_dict


def train_JEPA_batch(batch, masker, encoder, predictor, target_encoder,
                     optimizer, scheduler, wd_scheduler, momentum_scheduler,
                     device, epoch):
    
    #step the schedulers
    _new_lr = scheduler.step()
    _new_wd = wd_scheduler.step()
    momentum = next(momentum_scheduler)
    
    #prepare training
    encoder.train()
    predictor.train()
    target_encoder.eval()

    #send to device
    wave_batch = batch[0].float().to(device)
    masks_enc, masks_pred = masker(wave_batch)
    
    #1. forward pass:
    optimizer.zero_grad()
    #get encoder targets
    h = forward_target(target_encoder, wave_batch, masks_pred)
    #get context
    z = forward_context(encoder, predictor, 
                        wave_batch, masks_enc, masks_pred)
    #prediction loss
    loss, loss_dict = loss_fn(z, h)
    
    #2. backwards pass
    loss.backward()
    optimizer.step()
        
    #3. update the target encoder as weighted average with momentum
    with torch.no_grad():
        for param_q, param_k in zip(encoder.parameters(),
                                    target_encoder.parameters()):
            param_k.data.mul_(momentum).add_(
                (1.-momentum) * param_q.detach().data)
                
    loss_dict['epoch'] = epoch
    
    return loss_dict


@torch.no_grad()
def validate_JEPA_batch(batch, masker, encoder, predictor, target_encoder,
                     device, epoch):
    
    #prepare training
    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    #send to device
    wave_batch = batch[0].float().to(device)
    masks_enc, masks_pred = masker(wave_batch)
    
    #get encoder targets
    h = forward_target(target_encoder, wave_batch, masks_pred)
    #get context
    z = forward_context(encoder, predictor, 
                        wave_batch, masks_enc, masks_pred)
    #prediction loss
    loss, loss_dict = loss_fn(z, h)
           
    #record loss     
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


class ICBatchSampler(Sampler):
    def __init__(self, data_table, batch_size, num_samples):
        self.data_table = data_table
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.indices = self.create_batch_table_indices()
        
    def create_batch_table_indices(self):
        # Find unique elements in trial_ind
        unique_trials = self.data_table['trial_ind'].unique()
        
        # Initialize list to store sampled data
        sampled_data = []
        
        # Sample num_samples rows for each unique trial_ind
        for trial in unique_trials:
            trial_data = self.data_table[self.data_table['trial_ind'] == trial]
            sampled_data.append(trial_data.sample(n=self.num_samples, replace=True))
        
        # Concatenate all sampled data
        sampled_data = pd.concat(sampled_data)
        
        # Balance class_label by undersampling
        min_class_count = sampled_data['class_label'].value_counts().min()
        balanced_data = sampled_data.groupby('class_label').apply(lambda x: x.sample(min_class_count))
        
        # Shuffle the balanced_data
        batch_table = balanced_data.sample(frac=1)
        indices = batch_table.index.tolist()
        indicies = [i[1] for i in indices]
        
        return indicies
    
    def __iter__(self):
        # Get new set of indices
        self.indices = self.create_batch_table_indices()
        
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i + self.batch_size]
    
    def __len__(self):
        return len(self.indices)
    
    
class ICBatchSamplerRaw(Sampler):
    def __init__(self, data_table, batch_size):
        self.data_table = data_table
        self.batch_size = batch_size
        self.indices = self.create_batch_table_indices()
        
    def create_batch_table_indices(self):
        
        # Initialize list to store sampled data
        sampled_data = copy.deepcopy(self.data_table)
        
        # Balance class_label by undersampling
        min_class_count = sampled_data['class_label'].value_counts().min()
        balanced_data = sampled_data.groupby('class_label').apply(lambda x: x.sample(min_class_count))
        
        # Shuffle the balanced_data
        batch_table = balanced_data.sample(frac=1)
        indices = batch_table.index.tolist()
        indicies = [i[1] for i in indices]
        
        return indicies
    
    def __iter__(self):
        # Get new set of indices
        self.indices = self.create_batch_table_indices()
        
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i + self.batch_size]
    
    def __len__(self):
        return len(self.indices)


class ICBatchSamplerTrials(Sampler):
    def __init__(self, data_table, batch_size, weight=False):
        self.data_table = data_table
        self.batch_size = batch_size
        self.weight = weight
        self.indices = self.create_batch_table_indices()
        
    def create_batch_table_indices(self):
        
        # Initialize list to store sampled data
        sampled_data = copy.deepcopy(self.data_table)
        
        if self.weight:
            
            # Step 1: Oversample the Minority Class
            grouped_data = sampled_data.groupby('class_label')
            class_weights = grouped_data['weight'].sum()
            max_weight = class_weights.max()
            
            balanced_data = []
            
            for class_label, group in grouped_data:
                current_weight = class_weights[class_label]
                if current_weight < max_weight:
                    # Oversample to match the class with the maximum weight
                    oversample_group = group.sample(frac=(max_weight / current_weight), replace=True)
                else:
                    oversample_group = group
            
                balanced_data.append(oversample_group)
            
            # Step 2: Concatenate the oversampled data
            final_balanced_data = pd.concat(balanced_data)
            
            # Step 3: Calculate the weights of the classes
            final_weights_sums = final_balanced_data.groupby('class_label')['weight'].sum()
            
            iteration_limit = 10
            iterations = 0
            
            while (round(final_weights_sums[0]) != round(final_weights_sums[1])) and iterations < iteration_limit:
                iterations += 1
            
                # Identify the class with the larger weight
                larger_class_label = final_weights_sums.idxmax()
                smaller_class_label = final_weights_sums.idxmin()
                weight_difference = round(final_weights_sums[larger_class_label] - final_weights_sums[smaller_class_label])
            
                # Calculate the number of rows to remove to reduce the weight difference
                rows_to_remove = weight_difference // final_balanced_data.loc[
                    final_balanced_data['class_label'] == larger_class_label, 'weight'].min()
            
                # Ensure we remove at least one row to make progress
                rows_to_remove = max(1, rows_to_remove)
            
                # Remove the rows
                larger_class_data = final_balanced_data[final_balanced_data['class_label'] == larger_class_label]
                # Sort larger_class_data by the 'weight' column to ensure groups are ordered by weight
                sorted_larger_class_data = larger_class_data.sort_values(by='weight')
                
                # Group sorted_larger_class_data by weight
                grouped_larger_class_data = sorted_larger_class_data.groupby('weight')
                
                # Shuffle each group and stack them together
                shuffled_larger_class_data = pd.concat([group.sample(frac=1, replace=False) for _, group in grouped_larger_class_data])
                
                # Take the index of the first rows_to_remove as the removal index
                drop_indices = shuffled_larger_class_data.head(rows_to_remove).index
                final_balanced_data = final_balanced_data.drop(drop_indices)
            
                # Recalculate the weights after dropping
                final_weights_sums = final_balanced_data.groupby('class_label')['weight'].sum()
            
            # Step 5: Shuffle the balanced data
            batch_table = final_balanced_data.sample(frac=1)
            indices = batch_table.index.tolist()
        else:
            # Identify the majority class count
            max_class_count = sampled_data['class_label'].value_counts().max()
            
            # Separate data by class
            grouped_data = sampled_data.groupby('class_label')
            
            # Initialize list to collect oversampled data
            balanced_data = []
            
            for class_label, group in grouped_data:
                if len(group) < max_class_count:
                    # Oversample the minority class
                    sampled_group = group.sample(max_class_count, replace=True)
                else:
                    # Include all data from the majority class without replacement
                    sampled_group = group
                
                balanced_data.append(sampled_group)
            
            # Concatenate all balanced data
            balanced_data = pd.concat(balanced_data)
            
            # Shuffle the balanced data
            batch_table = balanced_data.sample(frac=1)
            indices = batch_table.index.tolist()
        
        return indices
    
    def __iter__(self):
        # Get new set of indices
        self.indices = self.create_batch_table_indices()
        
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i + self.batch_size]
    
    def __len__(self):
        return len(self.indices)


class TrainHistory:
    def __init__(self, keys):
        
        self._keys = keys
        self._epoch_dict = {}
        self._batch_dict = {}
        for d_type in ('train','val'):
            self._epoch_dict[d_type] = {key: [] for key in keys}
            self._batch_dict[d_type] = {key: [] for key in keys}
            
        #font properties
        font_path = r"C:\Windows\Fonts\Brink - BR Cobane Regular.otf"
        self.prop = fm.FontProperties(fname=font_path)
            
        #property of best loss
        self.best_loss = 0
        
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
                elif key == 'ROC_AUC':
                    data = self._batch_dict[d_type][key]

                    # Separate targets and prediction scores
                    target_list = [item[0] for item in data]
                    pred_list = [item[1] for item in data]
                    targets = np.concatenate(target_list)
                    preds = np.concatenate(pred_list)
                    
                    #calculate metric
                    if len(preds.shape) > 1:
                        probs = softmax(preds, axis=1)
                        if probs.shape[1] == 2:
                            probs = probs[:,1]
                            value = roc_auc_score(targets, probs)
                        else:
                            value = roc_auc_score(targets, probs,
                                                  multi_class='ovr',
                                                  average='macro')
                    else:
                        value = roc_auc_score(targets, preds)
                elif key == 'F1_SCORE':
                    data = self._batch_dict[d_type][key]

                    # Separate targets and prediction scores
                    target_list = [item[0] for item in data]
                    pred_list = [item[1] for item in data]
                    targets = np.concatenate(target_list)
                    preds = np.concatenate(pred_list)
                    
                    if len(preds.shape) > 1:
                        probs = softmax(preds, axis=1)
                        predicted_labels = np.argmax(probs, axis=1)
                        value = f1_score(targets, 
                                         predicted_labels,
                                         average='macro')
                    else:
                        #calculate metric
                        preds = preds > 0.5
                        value = f1_score(targets, preds)
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
            
            
#################### training loops for specific model types ##################

def train_pretrain_model_from_JEPA(cfg):
    
    #empty cache
    torch.cuda.empty_cache()
    
    #directory based parameters
    data_root = cfg['data_root']
    sbj_name = cfg['sbj_name']
    model_load_path = cfg['model_load_path']
    model_save_path = cfg['model_save_path']
    
    #convert to path
    data_root = Path(data_root)
    model_load_path = Path(model_load_path)
    model_save_path = Path(model_save_path)
    
    #model based parameters
    embed_dim = cfg['embed_dim']
    encoder_depth = cfg['encoder_depth']
    attn_drop_rate = cfg['attn_drop_rate']
    drop_rate = cfg['drop_rate']
    drop_path_rate = cfg['drop_path_rate']
    funnel_depth = cfg['funnel_depth']
    funnel_decimate = cfg['funnel_decimate']
    batch_size = cfg['batch_size']
    frozen_str = cfg['frozen_str']
    
    #reduce heads for micro
    if embed_dim == 32:
        num_heads = 8
    else:
        num_heads = 12
    
    #see if models exist
    history_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_history_final.pkl'
    if history_name.exists():
        print(f'{sbj_name} pretrained model exists; moving on...')
        return

    print(f'Running {frozen_str} Pretraining Model: {sbj_name}')
    
    #files that initialize datasets
    training_file = data_root / sbj_name / "sbj_traintest.csv"
    chan_file = data_root / sbj_name / "act_chans.csv"
    
    #get the training class proportions for each subject:
    sbj_table = pd.read_csv(training_file)
    chan_table = pd.read_csv(chan_file)
    num_chan = len(chan_table)
    num_classes = len(sbj_table['project_class'].unique())

    #load the data
    training_data = AllSbjData(sbj_table, chan_table, data_root, training=True)
    val_data = AllSbjData(sbj_table, chan_table, data_root, training=False, test_str = 'val')
    
    
    #prepare data loaders
    #make data sampler
    train_table = copy.deepcopy(training_data._packet_table)
    batch_sampler = ICBatchSamplerRaw(train_table, batch_size)
    
    #prepare data loaders
    train_dl = DataLoader(training_data, batch_sampler=batch_sampler)
    val_dl = DataLoader(val_data, batch_size=batch_size,
                          drop_last=False, shuffle = False)


    '''
    iter_dl = iter(train_dl)
    batch = next(iter_dl)
    
    wave = torch.squeeze(batch)
    
    '''
    
    #################### Load the models ##################################
    
    #load pre-trained JEPA_model state dict
    load_fn = model_load_path / f'{sbj_name}_JEPA_target_encoder_final.pth'
    state_dict = torch.load(load_fn)
    
    #initialize new predictor
    model = PretrainTransformer(
            wave_len=[512],
            patch_size=8,
            num_heads = num_heads,
            in_chans=num_chan,
            embed_dim=embed_dim,
            depth=encoder_depth,
            funnel_depth = funnel_depth,
            funnel_decimate = funnel_decimate,
            num_classes = num_classes,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate,
            )
    
    #transfer parameters and freeze layers
    copy_state_dict(state_dict, model)
    
    # Freeze the specified layers in WavePredictor
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    
    model.pos_embed.requires_grad = False
    
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False

    
    #send to device
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #################### get the optimizer ##################################
    
    nepochs = 300
    warmup = 15
    optimizer, scheduler, wd_scheduler = init_pretrain_opt(
        model,
        len(train_dl),
        start_lr=0.0001,
        ref_lr=0.001,
        warmup=warmup,
        num_epochs=nepochs,
        wd=0.04,
        final_wd=0.4,
        final_lr=1e-6,
        )
    
    
    #################### saving params ##################################
    
    log_keys = ['epoch',
                'CLASS_LOSS',
                'CLASS_ACCURACY',
                'ROC_AUC',
                'F1_SCORE']
    best_loss = 0.0
    
    history = TrainHistory(log_keys)
    class_weights = []
    nochange_epoch = 0
    for epoch in range(nepochs):
        print(f"Epoch {epoch} Running training loop...")

        for batch_idx, batch in tqdm(enumerate(train_dl)):
            train_dict =  train_pred_batch(batch, 
                                           model,
                                           optimizer,
                                           device,
                                           epoch,
                                           class_weights)
            history.update('train', train_dict)
            #step LR and WD
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
                  
        val_data.reset_seed()
        print(f"Epoch {epoch} Running validation loop...")
        for batch_idx, batch in tqdm(enumerate(val_dl)):
            val_dict = validate_pred_batch(batch,
                                           model,
                                           device,
                                           epoch,
                                           class_weights)
            history.update('val', val_dict)
                
        #evaluate model
        history.digest()
        val_loss = history.validation_value('ROC_AUC')
        if (val_loss > best_loss) & (epoch > warmup):
            #save model
            model_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_best.pth'
            torch.save(model.state_dict(), model_name)
            best_loss = val_loss
            #save history
            history_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_history_best.pkl'
            with open(history_name, 'wb') as file:
                pickle.dump(history, file)
            nochange_epoch = 0
        else:
            nochange_epoch += 1
            
        #unfreeze at epoch 100
        if (epoch == 100) & (frozen_str == "unfrozen"):
            model, optimizer, scheduler, wd_scheduler =  \
                unfreeze_model(model, train_dl)
            
                
    #save final model
    model_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_final.pth'
    torch.save(model.state_dict(), model_name)
    best_loss = val_loss
    #save history
    history_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_history_final.pkl'
    with open(history_name, 'wb') as file:
        pickle.dump(history, file)
    torch.cuda.empty_cache()        


def train_LDA_model_from_JEPA(cfg):
    
    #empty cache
    torch.cuda.empty_cache()
    
    #directory based parameters
    data_root = cfg['data_root']
    sbj_name = cfg['sbj_name']
    model_load_path = cfg['model_load_path']
    model_save_path = cfg['model_save_path']
    
    #convert to path
    data_root = Path(data_root)
    model_load_path = Path(model_load_path)
    model_save_path = Path(model_save_path)
    
    #model based parameters
    embed_dim = cfg['embed_dim']
    encoder_depth = cfg['encoder_depth']
    attn_drop_rate = cfg['attn_drop_rate']
    drop_rate = cfg['drop_rate']
    drop_path_rate = cfg['drop_path_rate']
    funnel_depth = cfg['funnel_depth']
    funnel_decimate = cfg['funnel_decimate']
    batch_size = cfg['batch_size']
    frozen_str = cfg['frozen_str']
    
    #reduce heads for micro
    if embed_dim == 32:
        num_heads = 8
    else:
        num_heads = 12
    
    #see if models exist
    history_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_history_final.pkl'
    if history_name.exists():
        print(f'{sbj_name} pretrained model exists; moving on...')
        return

    print(f'Running {frozen_str} Pretraining Model: {sbj_name}')
    
    #files that initialize datasets
    training_file = data_root / "training_table.csv"
    
    #get the training class proportions for each subject:
    train_table = pd.read_csv(training_file)
    sbj_table, num_chan, weights = get_sbj_table_pretrain(
        train_table, sbj_name, data_root)

    print(f'{sbj_name}: {len(weights)} classes')
    print(f'Class weights: {weights}')

    #load data
    training_data = AllSbjData(sbj_table, data_root, num_chan, training=True)
    val_data = AllSbjData(sbj_table, data_root, num_chan, training=False)
    
    
    #prepare data loaders
    #make data sampler
    train_table = copy.deepcopy(training_data._packet_table)
    batch_sampler = ICBatchSamplerRaw(train_table, batch_size)
    
    #prepare data loaders
    train_dl = DataLoader(training_data, batch_sampler=batch_sampler)
    val_dl = DataLoader(val_data, batch_size=batch_size,
                          drop_last=False, shuffle = False)


    '''
    iter_dl = iter(train_dl)
    batch = next(iter_dl)
    
    wave = torch.squeeze(batch)
    
    '''
    
    #################### Load the models ##################################
    
    #load pre-trained JEPA_model state dict
    load_fn = model_load_path / f'{sbj_name}_JEPA_target_encoder_final.pth'
    state_dict = torch.load(load_fn)
    
    #initialize new predictor
    model = ProbabilisticTransformer(
            wave_len=[512],
            patch_size=8,
            num_heads = num_heads,
            in_chans=num_chan,
            embed_dim=embed_dim,
            depth=encoder_depth,
            funnel_depth = funnel_depth,
            funnel_decimate = funnel_decimate,
            num_classes = len(weights),
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate,
            )
    
    #transfer parameters and freeze layers
    copy_state_dict(state_dict, model)
    
    # Freeze the specified layers in WavePredictor
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    
    model.pos_embed.requires_grad = False
    
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False

    
    #send to device
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #################### get the optimizer ##################################
    
    nepochs = 300
    warmup = 15
    optimizer, scheduler, wd_scheduler = init_pretrain_opt(
        model,
        len(train_dl),
        start_lr=0.0001,
        ref_lr=0.001,
        warmup=warmup,
        num_epochs=nepochs,
        wd=0.04,
        final_wd=0.4,
        final_lr=1e-6,
        )
    
    
    #################### saving params ##################################
    
    log_keys = ['epoch',
                'TOTAL_LOSS',
                'CLASS_LOSS',
                'KL_LOSS',
                'CLASS_ACCURACY',
                'ROC_AUC',
                'F1_SCORE']
    best_loss = 0.0
    
    history = TrainHistory(log_keys)
    class_weights = torch.tensor(weights.values, dtype=torch.float32).to(device)
    nochange_epoch = 0
    for epoch in range(nepochs):
        print(f"Epoch {epoch} Running training loop...")

        for batch_idx, batch in tqdm(enumerate(train_dl)):
            train_dict =  train_prob_batch(batch, 
                                           model,
                                           optimizer,
                                           device,
                                           epoch,
                                           class_weights)
            history.update('train', train_dict)
            #step LR and WD
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
                  
        val_data.reset_seed()
        print(f"Epoch {epoch} Running validation loop...")
        for batch_idx, batch in tqdm(enumerate(val_dl)):
            val_dict = validate_prob_batch(batch,
                                           model,
                                           device,
                                           epoch,
                                           class_weights)
            history.update('val', val_dict)
                
        #evaluate model
        history.digest()
        val_loss = history.validation_value('ROC_AUC')
        if (val_loss > best_loss) & (epoch > warmup):
            #save model
            model_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_best.pth'
            torch.save(model.state_dict(), model_name)
            best_loss = val_loss
            #save history
            history_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_history_best.pkl'
            with open(history_name, 'wb') as file:
                pickle.dump(history, file)
            nochange_epoch = 0
        else:
            nochange_epoch += 1
            
        #unfreeze at epoch 100
        if (epoch == 100) & (frozen_str == "unfrozen"):
            model, optimizer, scheduler, wd_scheduler =  \
                unfreeze_model(model, train_dl)
            
                
    #save final model
    model_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_final.pth'
    torch.save(model.state_dict(), model_name)
    best_loss = val_loss
    #save history
    history_name = model_save_path / f'{sbj_name}_pretrain_from_JEPA_history_final.pkl'
    with open(history_name, 'wb') as file:
        pickle.dump(history, file)
    torch.cuda.empty_cache()     


def train_JEPA_model(cfg):
    
    #empty cache
    torch.cuda.empty_cache()
    
    #directory based parameters
    data_root = cfg['data_root']
    sbj_name = cfg['sbj_name']
    model_save_path = cfg['model_save_path']
    
    #convert to path
    data_root = Path(data_root)
    model_save_path = Path(model_save_path)
    
    #check if computed
    history_name = model_save_path / f'{sbj_name}_JEPA_history_final.pkl'
    if history_name.exists():
        print(f'{sbj_name} JEPA model exists; moving on...')
        return
    
    #get model specific parameters
    nepochs = cfg['nepochs']
    embed_dim = cfg['embed_dim']
    encoder_depth = cfg['encoder_depth']
    attn_drop_rate = cfg['attn_drop_rate']
    drop_rate = cfg['drop_rate']
    batch_size = cfg['batch_size']
    
    #derive predictor as half
    predictor_embed_dim = np.max([int(embed_dim/2),32])
    predictor_depth = np.max([int(encoder_depth/2),2])
    
    #reduce heads for micro
    if predictor_embed_dim == 32:
        num_heads = 8
    else:
        num_heads = 12
    
    print(f'Running JEPA Pretraining Model: {sbj_name}')
    
    #data root where all info lives
    data_root = Path(data_root)
    
    #files that initialize datasets
    training_file = data_root / sbj_name / "sbj_traintest.csv"
    chan_file = data_root / sbj_name / "act_chans.csv"
    
    #get the training class proportions for each subject:
    sbj_table = pd.read_csv(training_file)
    chan_table = pd.read_csv(chan_file)
    num_chan = len(chan_table)

    #load the data
    training_data = AllSbjData(sbj_table, chan_table, data_root, training=True)
    val_data = AllSbjData(sbj_table, chan_table, data_root, training=False, test_str = 'val')
    
    
    #prepare data loaders
    train_dl = DataLoader(training_data, batch_size=batch_size,
                          drop_last=False, shuffle = True)
    val_dl = DataLoader(val_data, batch_size=batch_size,
                          drop_last=False, shuffle = False)

    '''
    iter_dl = iter(train_dl)
    batch = next(iter_dl)
    
    wave = torch.squeeze(batch)
    
    '''
    
    #mask collator
    train_masker = MaskCollatorWave(
            input_size=512,
            patch_size=8,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.05, 0.15),
            nenc=1,
            npred=4,
            training = True
            )
    
    val_masker = MaskCollatorWave(
            input_size=512,
            patch_size=8,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.05, 0.15),
            nenc=1,
            npred=4,
            training = False
            )
    
    
    #################### Load the models ##################################
    
    encoder = JEPAEncoder(
            wave_len=[512],
            patch_size=8,
            in_chans=num_chan,
            embed_dim=embed_dim,
            num_heads = num_heads,
            predictor_embed_dim=predictor_embed_dim,
            depth=encoder_depth,
            predictor_depth=predictor_depth,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            )
    
    num_patches=encoder.patch_embed.num_patches
    
    predictor = JEPAPredictor(
            num_patches,
            embed_dim=embed_dim,
            num_heads = num_heads,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            )
    
    target_encoder = copy.deepcopy(encoder)
    
    #send to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)
    
    
    
    #################### get the optimizer ##################################
    
    warmup = int((nepochs*15)/300)
    optimizer, scheduler, wd_scheduler = init_JEPA_opt(
        encoder,
        predictor,
        len(training_data),
        start_lr=0.0001,
        ref_lr=0.001,
        warmup=warmup,
        num_epochs=nepochs,
        wd=0.04,
        final_wd=0.4,
        final_lr=1e-6,
        )
    
    ema = (0.996, 1.0)
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(nepochs*len(training_data))
                              for i in range(int(nepochs*len(training_data))+1))
    
    
    #################### saving params ##################################
    
    model_save_path = Path(model_save_path)
    
    log_keys = ['epoch',
                'TOTAL_LOSS']
    
    history = TrainHistory(log_keys)
    for epoch in range(nepochs):
        print(f"Epoch {epoch} Running training loop...")
        
        #shuffle training data
        torch.cuda.empty_cache()
        for batch_idx, batch in tqdm(enumerate(train_dl)):
            train_dict =  train_JEPA_batch(batch,
                                           train_masker,
                                           encoder,
                                           predictor,
                                           target_encoder,
                                           optimizer,
                                           scheduler,
                                           wd_scheduler,
                                           momentum_scheduler,
                                           device,
                                           epoch)
            history.update('train', train_dict)
                  
        val_data.reset_seed()
        print(f"Epoch {epoch} Running validation loop...")
        for batch_idx, batch in tqdm(enumerate(val_dl)):
            val_dict = validate_JEPA_batch(batch,
                                           val_masker,
                                           encoder,
                                           predictor,
                                           target_encoder,
                                           device,
                                           epoch)
            history.update('val', val_dict)
                
        #evaluate model
        history.digest()
             
    #save final model
    #save the encoder
    model_name = model_save_path / f'{sbj_name}_JEPA_encoder_final.pth'
    torch.save(encoder.state_dict(), model_name)
    
    #save the predictor
    model_name = model_save_path / f'{sbj_name}_JEPA_predictor_final.pth'
    torch.save(predictor.state_dict(), model_name)
    
    #save the target_encoder
    model_name = model_save_path / f'{sbj_name}_JEPA_target_encoder_final.pth'
    torch.save(target_encoder.state_dict(), model_name)
    #save history
    history_name = model_save_path / f'{sbj_name}_JEPA_history_final.pkl'
    with open(history_name, 'wb') as file:
        pickle.dump(history, file)
        
    del encoder, predictor, target_encoder
    torch.cuda.empty_cache()
  
    
def train_cross_entropy_model(cfg):
    
    #empty cache
    torch.cuda.empty_cache()
    
    #directory based parameters
    data_root = cfg['data_root']
    sbj_name = cfg['sbj_name']
    model_save_path = cfg['model_save_path']
    
    #convert to path
    data_root = Path(data_root)
    model_save_path = Path(model_save_path)
    
    #model based parameters
    embed_dim = cfg['embed_dim']
    encoder_depth = cfg['encoder_depth']
    attn_drop_rate = cfg['attn_drop_rate']
    drop_rate = cfg['drop_rate']
    drop_path_rate = cfg['drop_path_rate']
    funnel_depth = cfg['funnel_depth']
    funnel_decimate = cfg['funnel_decimate']
    batch_size = cfg['batch_size']
    frozen_str = cfg['frozen_str']
    
    #reduce heads for micro
    if embed_dim == 32:
        num_heads = 8
    else:
        num_heads = 12
    
    #see if models exist
    history_name = model_save_path / f'{sbj_name}_pretrain_CE_history_final.pkl'
    if history_name.exists():
        print(f'{sbj_name} pretrained model exists; moving on...')
        return

    print(f'Running CE Pretraining Model: {sbj_name}')
    
    #files that initialize datasets
    training_file = data_root / sbj_name / "sbj_traintest.csv"
    chan_file = data_root / sbj_name / "act_chans.csv"
    
    #get the training class proportions for each subject:
    sbj_table = pd.read_csv(training_file)
    chan_table = pd.read_csv(chan_file)
    num_chan = len(chan_table)
    num_classes = len(sbj_table['project_class'].unique())

    #load the data
    training_data = AllSbjData(sbj_table, chan_table, data_root, training=True)
    val_data = AllSbjData(sbj_table, chan_table, data_root, training=False, test_str = 'val')
    
    
    #prepare data loaders
    #make data sampler
    train_table = copy.deepcopy(training_data._packet_table)
    batch_sampler = ICBatchSamplerRaw(train_table, batch_size)
    
    #prepare data loaders
    train_dl = DataLoader(training_data, batch_sampler=batch_sampler)
    val_dl = DataLoader(val_data, batch_size=batch_size,
                          drop_last=False, shuffle = False)


    '''
    iter_dl = iter(train_dl)
    batch = next(iter_dl)
    
    wave = torch.squeeze(batch)
    
    '''
    
    #################### Initialize model ##################################
    
    
    #initialize new predictor
    model = PretrainTransformer(
            wave_len=[512],
            patch_size=8,
            num_heads = num_heads,
            in_chans=num_chan,
            embed_dim=embed_dim,
            depth=encoder_depth,
            funnel_depth = funnel_depth,
            funnel_decimate = funnel_decimate,
            num_classes = num_classes,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate,
            )

    #send to device
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #################### get the optimizer ##################################
    
    nepochs = 300
    warmup = 15
    optimizer, scheduler, wd_scheduler = init_pretrain_opt(
        model,
        len(train_dl),
        start_lr=0.0001,
        ref_lr=0.001,
        warmup=warmup,
        num_epochs=nepochs,
        wd=0.04,
        final_wd=0.4,
        final_lr=1e-6,
        )
    
    
    #################### saving params ##################################
    
    log_keys = ['epoch',
                'CLASS_LOSS',
                'CLASS_ACCURACY',
                'ROC_AUC',
                'F1_SCORE']
    best_loss = 0.0
    
    history = TrainHistory(log_keys)
    class_weights = []
    nochange_epoch = 0
    for epoch in range(nepochs):
        print(f"Epoch {epoch} Running training loop...")

        for batch_idx, batch in tqdm(enumerate(train_dl)):
            train_dict =  train_pred_batch(batch, 
                                           model,
                                           optimizer,
                                           device,
                                           epoch,
                                           class_weights)
            history.update('train', train_dict)
            #step LR and WD
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
                  
        val_data.reset_seed()
        print(f"Epoch {epoch} Running validation loop...")
        for batch_idx, batch in tqdm(enumerate(val_dl)):
            val_dict = validate_pred_batch(batch,
                                           model,
                                           device,
                                           epoch,
                                           class_weights)
            history.update('val', val_dict)
                
        #evaluate model
        history.digest()
        val_loss = history.validation_value('ROC_AUC')
        if (val_loss > best_loss) & (epoch > warmup):
            #save model
            model_name = model_save_path / f'{sbj_name}_pretrain_CE_best.pth'
            torch.save(model.state_dict(), model_name)
            best_loss = val_loss
            #save history
            history_name = model_save_path / f'{sbj_name}_pretrain_CE_history_best.pkl'
            with open(history_name, 'wb') as file:
                pickle.dump(history, file)
            nochange_epoch = 0
        else:
            nochange_epoch += 1
            
            
                
    #save final model
    model_name = model_save_path / f'{sbj_name}_pretrain_CE_final.pth'
    torch.save(model.state_dict(), model_name)
    best_loss = val_loss
    #save history
    history_name = model_save_path / f'{sbj_name}_pretrain_CE_history_final.pkl'
    with open(history_name, 'wb') as file:
        pickle.dump(history, file)
    torch.cuda.empty_cache()        
    
    
    
def test_pretrained_model(cfg):
    
    #empty cache
    torch.cuda.empty_cache()
    
    #directory based parameters
    data_root = cfg['data_root']
    sbj_name = cfg['sbj_name']
    model_load_path = cfg['model_load_path']
    
    #convert to path
    data_root = Path(data_root)
    model_load_path = Path(model_load_path)
    
    #model based parameters
    embed_dim = cfg['embed_dim']
    encoder_depth = cfg['encoder_depth']
    attn_drop_rate = cfg['attn_drop_rate']
    drop_rate = cfg['drop_rate']
    drop_path_rate = cfg['drop_path_rate']
    funnel_depth = cfg['funnel_depth']
    funnel_decimate = cfg['funnel_decimate']
    batch_size = cfg['batch_size']
    
    #reduce heads for micro
    if embed_dim == 32:
        num_heads = 8
    else:
        num_heads = 12
    
    #get model name
    if cfg['model_type'] == 'JEPA':
        model_name = model_load_path / f'{sbj_name}_pretrain_from_JEPA_best.pth'
    elif cfg['model_type'] == 'CE':
        model_name = model_load_path / f'{sbj_name}_pretrain_CE_best.pth'
    
    #files that initialize datasets
    training_file = data_root / sbj_name / "sbj_traintest.csv"
    chan_file = data_root / sbj_name / "act_chans.csv"
    
    #get the training class proportions for each subject:
    sbj_table = pd.read_csv(training_file)
    chan_table = pd.read_csv(chan_file)
    num_chan = len(chan_table)
    num_classes = len(sbj_table['project_class'].unique())

    #load the data
    test_data = AllSbjData(sbj_table, chan_table, data_root, training=False, test_str = 'test')
    
    test_dl = DataLoader(test_data, batch_size=batch_size,
                          drop_last=False, shuffle = False)


    '''
    iter_dl = iter(test_dl)
    batch = next(iter_dl)
    
    wave = torch.squeeze(batch)
    
    '''
    
    
    #################### Load the models ##################################
    
    #load pre-trained JEPA_model state dict
    state_dict = torch.load(model_name)
    
    #initialize new predictor
    model = PretrainTransformer(
            wave_len=[512],
            patch_size=8,
            num_heads = num_heads,
            in_chans=num_chan,
            embed_dim=embed_dim,
            depth=encoder_depth,
            funnel_depth = funnel_depth,
            funnel_decimate = funnel_decimate,
            num_classes = num_classes,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate,
            )
    
    #transfer parameters and freeze layers
    copy_state_dict(state_dict, model)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.training = False
    model.eval()

    #send to device
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #################### initialize data ##################################
    
    model_predictions = np.zeros((len(test_data),num_classes))
    class_labels = np.zeros(len(test_data))
    curr_ind = 0
    
    
    ########### predict in batches #########################################

    for batch_idx, batch in tqdm(enumerate(test_dl)):
        #pull out batch data
        wave, batch_labels = batch
        
        #send to device
        wave = wave.float().to(device)

        #prepare training
        model.eval()
        #forward pass
        logits = model(wave)
        
        #current size
        B = logits.shape[0]
        
        #get raw predictions
        probs_raw = F.softmax(logits, dim=1)
        probs_raw = probs_raw.detach().cpu().numpy()
        model_predictions[curr_ind:(curr_ind+B)] = probs_raw
        
        #add label and update
        class_labels[curr_ind:(curr_ind+B)] = batch_labels.numpy()
        curr_ind = curr_ind + B
        
    
    #evaluate the model
    # Convert probabilities to binary predictions (assuming 0.5 as the threshold)
    class_predictions = np.argmax(model_predictions,axis=1).astype(int)
    f1 = f1_score(class_labels, class_predictions, average='weighted')
    
    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(class_labels, model_predictions,
                          multi_class='ovr',
                          average='macro')
    
    #get classification accuracy
    correct = class_predictions == class_labels
    ncorrect = np.sum(correct)
    acc = ncorrect/len(class_labels)
    
    #calculate ROC curve
    fpr, tpr = get_average_roc_curves(class_labels, model_predictions)
    
    model_dict = {
        'num_classes': num_classes,
        'f1': f1,
        'auc': roc_auc,
        'acc': acc,
        'ROC_x': fpr, 
        'ROC_y': tpr, 
        }
    
    return model_dict
     
    
    
def get_average_roc_curves(class_labels, model_predictions):
    # Assuming `class_labels` and `model_predictions` are provided
    n_classes = len(np.unique(class_labels))  # Number of classes
    class_labels_binarized = label_binarize(class_labels, classes=np.arange(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # For macro-average ROC curve computation
    all_fpr = np.unique(np.concatenate([roc_curve(class_labels_binarized[:, i], model_predictions[:, i])[0] for i in range(n_classes)]))
    
    # Initialize mean TPR (True Positive Rate)
    mean_tpr = np.zeros_like(all_fpr)
    
    # Loop through each class to compute fpr, tpr and ROC AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(class_labels_binarized[:, i], model_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Interpolate the TPR values at points in `all_fpr` and add to mean_tpr
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Average the TPR across all classes
    mean_tpr /= n_classes
    
    # interpolate for subject averaging
    num_points = 1000
    fpr_interp = np.linspace(all_fpr.min(), all_fpr.max(), num_points)
    tpr_interp = interp1d(all_fpr, mean_tpr)(fpr_interp)  # Interpolate TPR to match FPR points
    
    return fpr_interp,  tpr_interp
    