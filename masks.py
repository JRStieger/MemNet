# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:49:21 2024

@author: jstieger
"""

#functions to generate wave masks
import torch
import time

    
class MaskCollatorWave(object):
    
    def __init__(
        self,
        input_size=512,
        patch_size=8,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.05, 0.15),
        nenc=1,
        npred=4,
        training = True
    ):
        super(MaskCollatorWave, self).__init__()
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        if training:
            self.seed = int(time.time())
        else:
            self.seed = int(42)
        self.generator = torch.Generator()
        self.reset_seed()
        
    def reset_seed(self):
        self.generator.manual_seed(self.seed)
        
    def _sample_block_size(self, scale):
        _rand = torch.rand(1, generator=self.generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.num_patches * mask_scale)

        return max_keep

    def _sample_block_mask(self, m_size):
        
        #get random start index and takes consecutive time frames
        _rand = torch.rand(1, generator=self.generator).item()
        # -- get the maximum start index
        max_s = self.num_patches - m_size
        rand_start = int(_rand * (max_s))
        block_mask = torch.arange(rand_start, rand_start + m_size)
        
        return block_mask

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)
        device = batch.device

        p_size = self._sample_block_size(
            scale=self.pred_mask_scale)
        e_size = self._sample_block_size(
            scale=self.enc_mask_scale)
        
        #get prediction masks
        masks_p = []
        for _ in range(self.npred):
            mask = self._sample_block_mask(p_size)
            masks_p.append(mask)
            
        #find the union over prediction mask elements
        unique_pred = torch.unique(torch.cat(masks_p))
        
        #find the encoding mask
        mask_e = self._sample_block_mask(e_size)
        rm_mask = ~torch.isin(mask_e, unique_pred)
        mask_enc = mask_e[rm_mask]
        
        #repeat across batch entries
        collated_masks_enc = [mask_enc.unsqueeze(0).repeat(B, 1).to(device)]
        collated_masks_pred = [mp.unsqueeze(0).repeat(B, 1).to(device)
                               for mp in masks_p]

        return collated_masks_enc, collated_masks_pred
    
    
def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1)).to(x.device)
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0).to(x.device)