# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:56:55 2024

Module contains main dataset structures classes for the memnet package

@author: jstieger
"""

import copy
import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset
import time


class AllSbjData(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, sbj_table, chan_table, data_root,
                 training=True, test_str = 'val',
                 pkt_size = 512, step_size = 100):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self.training = training
        self._data_table = sbj_table[
            sbj_table.training == training].reset_index(drop=True)
        self._chan_table = chan_table
        self.test_str = test_str
        self.data_root = data_root
        self.num_chan = len(chan_table)
        self.pkt_size = pkt_size
        self.step_size = step_size
        #load all the subject's data
        self._load_data()
        #prepare generators
        if training:
            self.seed = int(time.time())
        else:
            self.seed = int(42)
        self.generator = torch.Generator()
        self.reset_seed()
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
        
    
    def _load_data(self):
        
        sbj_table = self._data_table
        chan_table = self._chan_table
        wave_data = []
        paket_rows = []
        #loop through and aload all files
        for idx in range(len(sbj_table)):
            sbj_name = sbj_table["subject"][idx]
            project = sbj_table["project"][idx]
            block_num = sbj_table["block_num"][idx]
            class_label = sbj_table["project_class"][idx]
            mat_folder = self.data_root / sbj_name / project / f'block_{block_num}'
            print(f'Loading {sbj_name} file {idx} of {len(sbj_table)}')
            
            #load the data
            for chan_idx in range(len(chan_table)):
                chan = chan_table['chan_inds'][chan_idx]
                mat_fn = mat_folder / f'wavedata_chan_{chan}.mat'
                matfile = sio.loadmat(mat_fn)
                wave = np.squeeze(matfile["wave"])
                if chan_idx == 0:
                    n_time = len(wave)
                    file_wave = np.zeros((self.num_chan, n_time))
                file_wave[chan_idx] = wave
                
            #calculate the number of packets
            pkt_size = self.pkt_size
            num_windows = (n_time // pkt_size) - 1
            
            #add data
            wave_data.append(file_wave)
            
            #create packet rows
            file_rows = [{'data_idx': idx,
                          'pkt_num': i,
                          'start_ind': i*self.pkt_size,
                          'class_label': class_label}
                         for i in range(num_windows)]
            
            #split training and testing
            if not self.training:
                packet_len = len(file_rows)
                if self.test_str == 'val':
                    #take first half of recording
                    file_rows = file_rows[:round(packet_len/2)]
                elif self.test_str == 'test':
                    #take first half of recording
                    file_rows = file_rows[round(packet_len/2):]
            #add to packet table
            paket_rows.extend(file_rows)
            
        #add data to self
        self._packet_table = pd.DataFrame(paket_rows)
        self._wave_data = wave_data
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["data_idx"][idx]
        class_label = torch.tensor(self._packet_table["class_label"][idx])
        start_ind = self._packet_table["start_ind"][idx]
        
        #random start for training data
        if self.training:
            #get the starting index
            _rand = torch.rand(1, generator=self.generator).item()
            max_start = self.pkt_size - 1
            rand_start = int(_rand * max_start)
        else:
            rand_start = 0
            
        #pull out the packets
        packet = self._wave_data[data_idx][:,(rand_start + start_ind):
                             (rand_start + (start_ind + self.pkt_size))]
        
        #normalize voltage distributions
        one_wave = self.normalize(packet)
        
        return one_wave, class_label
            
            
    def normalize(self, packet):
        
        #calc meta features
        mean = np.mean(packet)
        std = np.std(packet)
        
        #normalize packet
        packet_norm = (packet - mean)/(std + np.finfo(float).eps)
        
        #find and clip at 99th percentile
        max_clip = np.percentile(np.abs(packet_norm),98.4)
        packet_norm = np.clip(packet_norm,
                              a_min=-1*max_clip,
                              a_max=max_clip)
        #set maximum value to 1
        packet_norm = packet_norm/(
            max_clip + np.finfo(float).eps)
        packet_norm = torch.tensor(packet_norm)
        
        return packet_norm

    def reset_seed(self):
        self.generator.manual_seed(self.seed)
        
    def shuffle_packet_table(self):
        '''
        function shuffles observations in the packet table
        keeping the structure of files together in order

        Returns
        -------
        None.

        '''
        # Step 1: Split the table into tables that have the same number in the data_idx column
        grouped_tables = [group for _, group in 
                          self._packet_table.groupby('data_idx')]
        
        # Step 2: Shuffle the rows within each of these tables
        shuffled_tables = [table.sample(frac=1).reset_index(drop=True)
                           for table in grouped_tables]
        
        # Step 3: Shuffle the order of these tables and stack them
        np.random.shuffle(shuffled_tables)
        self._packet_table = pd.concat(shuffled_tables).reset_index(drop=True)
        
        
        
class ICDataRaw(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, sbj_table, chan_table, data_root,
                 pkt_size = 512, step_size=1):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self.data_root = data_root
        self._chan_table = chan_table
        self.num_chan = len(chan_table)
        self.pkt_size = pkt_size
        self.step_size = step_size
        
        #add data table
        self._data_table = sbj_table
        
        #load all the subject's data
        self._load_data()
        
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
        
    
    def _load_data(self):
        
        sbj_table = self._data_table
        chan_table = self._chan_table
        wave_data = []
        paket_rows = []
        #loop through and aload all files
        for idx in range(len(sbj_table)):
            sbj_name = sbj_table["subject"][idx]
            block_num = sbj_table["block_num"][idx]
            mat_folder = self.data_root / sbj_name / f'block_{block_num}'
            print(f'Loading {sbj_name} file {idx} of {len(sbj_table)}')
                
            #load the data
            for chan_idx in range(len(chan_table)):
                chan = chan_table['chan_inds'][chan_idx]
                mat_fn = mat_folder / f'wavedata_chan_{chan}.mat'
                matfile = sio.loadmat(mat_fn)
                wave = np.squeeze(matfile["wave"])
                if chan_idx == 0:
                    n_time = len(wave)
                    file_wave = np.zeros((self.num_chan, n_time))
                    time = np.squeeze(matfile["time"])
                file_wave[chan_idx] = wave
            # Calculate the number of windows
            pkt_size = self.pkt_size
            step_size = self.step_size
            num_windows = (n_time - pkt_size) // step_size + 1
            
            # Add data
            wave_data.append(file_wave)
        
            # Create packet rows
            for i in range(num_windows):
                start_ind = i * step_size
                end_ind = start_ind + pkt_size
                avg_time = np.mean(time[start_ind:end_ind])
                
                file_rows = {'data_idx': idx,  # assuming a single data index for this example
                             'pkt_num': i,
                             'start_ind': start_ind,
                             'time': avg_time}
                paket_rows.append(file_rows)
            
        #add data to self
        self._packet_table = pd.DataFrame(paket_rows)
        self._wave_data = wave_data
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["data_idx"][idx]
        start_ind = self._packet_table["start_ind"][idx]
            
        #pull out the packets
        packet = self._wave_data[data_idx][:,start_ind:
                             (start_ind + self.pkt_size)]
        
        #normalize voltage distributions
        one_wave = self.normalize(packet)
        
        return one_wave
            
            
    def normalize(self, packet):
        
        #calc meta features
        mean = np.mean(packet)
        std = np.std(packet)
        
        #normalize packet
        packet_norm = (packet - mean)/(std + np.finfo(float).eps)
        
        #find and clip at 99th percentile
        max_clip = np.percentile(np.abs(packet_norm),98.4)
        packet_norm = np.clip(packet_norm,
                              a_min=-1*max_clip,
                              a_max=max_clip)
        #set maximum value to 1
        packet_norm = packet_norm/(
            max_clip + np.finfo(float).eps)
        packet_norm = torch.tensor(packet_norm)
        
        return packet_norm
    
    
class CCEPDataRaw(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, chan_table, data_root,
                 pkt_size = 512, step_size=5):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self.data_root = data_root
        self._chan_table = chan_table
        self.num_chan = len(chan_table)
        self.pkt_size = pkt_size
        self.step_size = step_size
        
        #load all the subject's data
        self._load_data()
        
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
        
    
    def _load_data(self):
        
        chan_table = self._chan_table
        wave_data = []
        paket_rows = []
            
        #load the data
        for chan_idx in range(len(chan_table)):
            chan = chan_table['chan_inds'][chan_idx]
            mat_fn = self.data_root / f'wavedata_chan_{chan}.mat'
            matfile = sio.loadmat(mat_fn)
            wave = np.squeeze(matfile["wave"])
            if chan_idx == 0:
                n_time = len(wave)
                file_wave = np.zeros((self.num_chan, n_time))
                time = np.squeeze(matfile["time"])
            file_wave[chan_idx] = wave
        # Calculate the number of windows
        pkt_size = self.pkt_size
        step_size = self.step_size
        num_windows = (n_time - pkt_size) // step_size + 1
        
        # Add data
        wave_data.append(file_wave)
    
        # Create packet rows
        for i in range(num_windows):
            start_ind = i * step_size
            end_ind = start_ind + pkt_size
            avg_time = np.mean(time[start_ind:end_ind])
            
            file_rows = {'data_idx': 0,  # assuming a single data index for this example
                         'pkt_num': i,
                         'start_ind': start_ind,
                         'time': avg_time}
            paket_rows.append(file_rows)
            
        #add data to self
        self._packet_table = pd.DataFrame(paket_rows)
        self._wave_data = wave_data
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["data_idx"][idx]
        start_ind = self._packet_table["start_ind"][idx]
            
        #pull out the packets
        packet = self._wave_data[data_idx][:,start_ind:
                             (start_ind + self.pkt_size)]
        
        #normalize voltage distributions
        one_wave = self.normalize(packet)
        
        return one_wave
            
            
    def normalize(self, packet):
        
        #calc meta features
        mean = np.mean(packet)
        std = np.std(packet)
        
        #normalize packet
        packet_norm = (packet - mean)/(std + np.finfo(float).eps)
        
        #find and clip at 99th percentile
        max_clip = np.percentile(np.abs(packet_norm),98.4)
        packet_norm = np.clip(packet_norm,
                              a_min=-1*max_clip,
                              a_max=max_clip)
        #set maximum value to 1
        packet_norm = packet_norm/(
            max_clip + np.finfo(float).eps)
        packet_norm = torch.tensor(packet_norm)
        
        return packet_norm
    
class CCEPDataFeat(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, all_features, packet_table):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self.features = all_features
        self._packet_table = packet_table
        
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
            
    def __getitem__(self, idx):
        #normalize voltage distributions
        one_wave = self.features[idx]
        return one_wave
            
   
    
class ICDataFeatures(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, trialinfo, data_root, training):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self.training = training
        self._data_table = trialinfo[
            trialinfo.training == training].reset_index(drop=True)
        self.data_root = data_root / 'trial'
        
        #load all the subject's data
        self._load_data()
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
        
    
    def _load_data(self):
        
        data_table = self._data_table
        wave_data = []
        paket_rows = []
        #loop through and aload all files
        for index, trial_row in data_table.iterrows():
            trial_ind = trial_row['relative_trial']
            
            # Load the trial_dict from the pickle file
            trial_file = self.data_root / f'trial_{trial_ind}.pkl'
            with open(trial_file, 'rb') as file:
                trial_dict = pickle.load(file)
            
            #pull out the data
            wave = trial_dict['wave']
            wave_data.append(wave)
            
            #make the packet table for this trial
            wave_len = len(wave)
            trial_row['trial_ind'] = index
            repeated_rows = pd.DataFrame([trial_row] * wave_len)
            repeated_rows['time_ind'] = range(wave_len)
            paket_rows.append(repeated_rows)
            
        #add data to self
        self._packet_table = pd.concat(paket_rows, ignore_index=True)
        self._wave_data = wave_data
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["trial_ind"][idx]
        time_ind = self._packet_table["time_ind"][idx]
        class_label = self._packet_table["class_label"][idx]  
        
        #pull out the data
        packet = self._wave_data[data_idx][time_ind]

        return packet, class_label
            
    def get_class_weights(self):
        class_labels = self._packet_table['class_label']
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(class_labels),
                                             y=class_labels)
        return torch.tensor(class_weights, dtype=torch.float32)
    
    
class ICDataRawTrials(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, trialinfo, data_root, pkt_size=512, shift = 32):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self._data_table = copy.deepcopy(trialinfo.reset_index(drop=True))
        self.data_root = data_root / 'trial'
        self.pkt_size = pkt_size
        self.shift = shift
        
        #load all the subject's data
        self._load_data()
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
       
    def training_data(self, training):
        """
        method downsamples pkt_table to traing or testing 

        Parameters
        ----------
        training : training bool

        Returns
        -------
        None.

        """
        self.training = training
        self._packet_table = self._packet_table[
            self._packet_table.training == training].reset_index(drop=True)
        
    
    def _load_data(self):
        
        data_table = self._data_table
        wave_data = []
        paket_rows = []
        #loop through and aload all files
        for index, trial_row in data_table.iterrows():
            trial_ind = trial_row['relative_trial']
            
            # Load the trial_dict from the pickle file
            trial_file = self.data_root / f'trial_{trial_ind}.pkl'
            with open(trial_file, 'rb') as file:
                trial_dict = pickle.load(file)
            
            #pull out the data
            wave = trial_dict['wave']
            wave_data.append(wave)
            
            #make the packet table for this trial
            wave_len = wave.shape[1] - (self.pkt_size + self.shift)
            trial_row['trial_ind'] = index
            repeated_rows = pd.DataFrame([trial_row] * wave_len)
            repeated_rows['time_ind'] = range(wave_len)
            
            #get test percentage
            repeated_rows['training'] = True
            test_prct = round(wave_len/5)
            test_max = wave_len - test_prct
            test_start = np.random.randint(0, test_max)
            test_end = test_start + test_prct
            repeated_rows.iloc[test_start:test_end, repeated_rows.columns.get_loc('training')] = False
            
            #downsample
            repeated_rows = repeated_rows.iloc[::self.shift]
            
            #add to table
            paket_rows.append(repeated_rows)
            
        #add data to self
        self._packet_table = pd.concat(paket_rows, ignore_index=True)
        self._wave_data = wave_data
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["trial_ind"][idx]
        start_ind = self._packet_table["time_ind"][idx]
        class_label = self._packet_table["class_label"][idx]  
        
        #random start for training data
        if self.training:
            #get the starting index
            start_ind = start_ind + np.random.randint(0, self.shift)
            
        #pull out the packets
        packet = self._wave_data[data_idx][:,start_ind:
                             (start_ind + self.pkt_size)]
        
        #normalize voltage distributions
        one_wave = self.normalize(packet)
        
        return one_wave, class_label
            
            
    def normalize(self, packet):
        
        #calc meta features
        mean = np.mean(packet)
        std = np.std(packet)
        
        #normalize packet
        packet_norm = (packet - mean)/(std + np.finfo(float).eps)
        
        #find and clip at 99th percentile
        max_clip = np.percentile(np.abs(packet_norm),98.4)
        packet_norm = np.clip(packet_norm,
                              a_min=-1*max_clip,
                              a_max=max_clip)
        #set maximum value to 1
        packet_norm = packet_norm/(
            max_clip + np.finfo(float).eps)
        packet_norm = torch.tensor(packet_norm)
        
        return packet_norm  
    
    
class ICDataAttnTrials(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self,
                 trialinfo,
                 data_root,
                 shift = 5,
                 weight=False,
                 data_type='trial'):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self._data_table = copy.deepcopy(trialinfo.reset_index(drop=True))
        self.data_root = data_root / data_type
        self.shift = shift
        self.weight = weight
        
        #load all the subject's data
        self._load_data()
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
       
    def training_data(self, training):
        """
        method downsamples pkt_table to traing or testing 

        Parameters
        ----------
        training : training bool

        Returns
        -------
        None.

        """
        if training == 'test':
            self.training = False
        else:
            self.training = training
            self._packet_table = self._packet_table[
                self._packet_table.training == training].reset_index(drop=True)
        
    
    def _load_data(self):
        
        data_table = self._data_table
        wave_data = []
        abs_time = []
        rel_time = []
        paket_rows = []
        #loop through and aload all files
        for index, trial_row in data_table.iterrows():
            trial_ind = trial_row['relative_trial']
            
            # Load the trial_dict from the pickle file
            trial_file = self.data_root / f'trial_{trial_ind}.pkl'
            with open(trial_file, 'rb') as file:
                trial_dict = pickle.load(file)
            
            #pull out the data
            wave = trial_dict['wave']
            wave_data.append(wave)
            
            #add time
            abs_time.append(trial_dict['abs_time'])
            rel_time.append(trial_dict['rel_time'])
            
            #make the packet table for this trial
            wave_len = wave.shape[0] - self.shift
            if wave_len <= 1000:
                val_pnts = round(0.4*wave_len)
            else:
                val_pnts = 512 + round(0.1*wave_len)
                
            trial_row['trial_ind'] = index
            repeated_rows = pd.DataFrame([trial_row] * wave_len)
            repeated_rows['time_ind'] = range(wave_len)
            repeated_rows['weight'] = repeated_rows['conf_bool'] + 1
            
            #get test percentage
            repeated_rows['training'] = True
            test_max = wave_len - val_pnts
            test_start = np.random.randint(0, test_max)
            test_end = test_start + val_pnts
            repeated_rows.iloc[test_start:test_end, repeated_rows.columns.get_loc('training')] = False
            
            #downsample
            if self.shift > 1:
                repeated_rows = repeated_rows.iloc[::self.shift]
            
            #add to table
            paket_rows.append(repeated_rows)
            
        #add data to self
        self._packet_table = pd.concat(paket_rows, ignore_index=True)
        self._wave_data = wave_data
        self._abs_time = abs_time
        self._rel_time = rel_time
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["trial_ind"][idx]
        start_ind = self._packet_table["time_ind"][idx]
        class_label = self._packet_table["class_label"][idx]  
        
        #random start for training data
        if self.training:
            #get the starting index
            start_ind = start_ind + np.random.randint(0, self.shift)
            
        #pull out the packets
        packet = self._wave_data[data_idx][start_ind]
        
        if self.weight:
            weight = self._packet_table["weight"][idx]  
            return packet, class_label, weight
        else:
            return packet, class_label
    
    def get_time(self,trial):
        trial_table = self._packet_table.loc[
            self._packet_table['trial_ind']==trial]
        time_ind = trial_table['time_ind']
        abs_time = self._abs_time[trial][time_ind]
        rel_time = self._rel_time[trial][time_ind]
        return abs_time, rel_time
            
    
    
class ICDataAttnTrials_RAW(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self,
                 trialinfo,
                 data_root,
                 shift = 5,
                 pkt_size = 512,
                 weight=False,
                 data_type='trial'):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self._data_table = copy.deepcopy(trialinfo.reset_index(drop=True))
        self.data_root = data_root / data_type
        self.shift = shift
        self.weight = weight
        self.pkt_size = 512
        
        #load all the subject's data
        self._load_data()
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
       
    def training_data(self, training):
        """
        method downsamples pkt_table to traing or testing 

        Parameters
        ----------
        training : training bool

        Returns
        -------
        None.

        """
        if training == 'test':
            self.training = False
        else:
            self.training = training
            self._packet_table = self._packet_table[
                self._packet_table.training == training].reset_index(drop=True)
        
    
    def _load_data(self):
        
        data_table = self._data_table
        wave_data = []
        abs_time = []
        rel_time = []
        paket_rows = []
        #loop through and aload all files
        for index, trial_row in data_table.iterrows():
            trial_ind = trial_row['relative_trial']
            
            # Load the trial_dict from the pickle file
            trial_file = self.data_root / f'trial_{trial_ind}.pkl'
            with open(trial_file, 'rb') as file:
                trial_dict = pickle.load(file)
            
            #pull out the data
            wave = trial_dict['wave']
            wave_data.append(wave)
            
            #add time
            abs_time.append(trial_dict['abs_time'])
            rel_time.append(trial_dict['rel_time'])
            
            #make the packet table for this trial
            wave_len = wave.shape[1] - self.shift - self.pkt_size
            if wave_len <= 1000:
                val_pnts = round(0.4*wave_len)
            else:
                val_pnts = 512 + round(0.1*wave_len)
                
            trial_row['trial_ind'] = index
            repeated_rows = pd.DataFrame([trial_row] * wave_len)
            repeated_rows['time_ind'] = range(wave_len)
            repeated_rows['weight'] = repeated_rows['conf_bool'] + 1
            
            #get test percentage
            repeated_rows['training'] = True
            test_max = wave_len - val_pnts
            test_start = np.random.randint(0, test_max)
            test_end = test_start + val_pnts
            repeated_rows.iloc[test_start:test_end, repeated_rows.columns.get_loc('training')] = False
            
            #downsample
            if self.shift > 1:
                repeated_rows = repeated_rows.iloc[::self.shift]
            
            #add to table
            paket_rows.append(repeated_rows)
            
        #add data to self
        self._packet_table = pd.concat(paket_rows, ignore_index=True)
        self._wave_data = wave_data
        self._abs_time = abs_time
        self._rel_time = rel_time
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["trial_ind"][idx]
        start_ind = self._packet_table["time_ind"][idx]
        class_label = self._packet_table["class_label"][idx]  
        
        #random start for training data
        if self.training:
            #get the starting index
            start_ind = start_ind + np.random.randint(0, self.shift)
            
        #pull out the packets
        packet = self._wave_data[data_idx][:,start_ind:
                             (start_ind + self.pkt_size)]
        
        #normalize voltage distributions
        one_wave = self.normalize(packet)
        
        if self.weight:
            weight = self._packet_table["weight"][idx]  
            return one_wave, class_label, weight
        else:
            return one_wave, class_label
                
    def normalize(self, packet):
        
        #calc meta features
        mean = np.mean(packet)
        std = np.std(packet)
        
        #normalize packet
        packet_norm = (packet - mean)/(std + np.finfo(float).eps)
        
        #find and clip at 99th percentile
        max_clip = np.percentile(np.abs(packet_norm),98.4)
        packet_norm = np.clip(packet_norm,
                              a_min=-1*max_clip,
                              a_max=max_clip)
        #set maximum value to 1
        packet_norm = packet_norm/(
            max_clip + np.finfo(float).eps)
        packet_norm = torch.tensor(packet_norm)
        
        return packet_norm

    def get_time(self,trial):
        trial_table = self._packet_table.loc[
            self._packet_table['trial_ind']==trial]
        time_ind = trial_table['time_ind']
        abs_time = self._abs_time[trial][time_ind]
        rel_time = self._rel_time[trial][time_ind]
        return abs_time, rel_time
    
    
    
class ICDataAttn(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, trialinfo, data_root, training, pkt_size = 50):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self.training = training
        self._data_table = trialinfo[
            trialinfo.training == training].reset_index(drop=True)
        self.data_root = data_root / 'trial'
        self.pkt_size = pkt_size
        
        #load all the subject's data
        self._load_data()
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
        
    
    def _load_data(self):
        
        data_table = self._data_table
        wave_data = []
        paket_rows = []
        #loop through and aload all files
        for index, trial_row in data_table.iterrows():
            trial_ind = trial_row['relative_trial']
            
            # Load the trial_dict from the pickle file
            trial_file = self.data_root / f'trial_{trial_ind}.pkl'
            with open(trial_file, 'rb') as file:
                trial_dict = pickle.load(file)
            
            #pull out the data
            wave = trial_dict['wave']
            wave_data.append(wave)
            
            #make the packet table for this trial

            #make the packet table for this trial
            wave_len = wave.shape[0]
            pkt_size = self.pkt_size
            step_size = 1
            num_windows = (wave_len - pkt_size) // step_size
            wave_start = np.arange(num_windows)*step_size
            
            trial_row['trial_ind'] = index
            repeated_rows = pd.DataFrame([trial_row] * num_windows)
            repeated_rows['start_ind'] = wave_start
            paket_rows.append(repeated_rows)

        #add data to self
        self._packet_table = pd.concat(paket_rows, ignore_index=True)
        self._wave_data = wave_data
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["trial_ind"][idx]
        time_ind = self._packet_table["start_ind"][idx]
        class_label = self._packet_table["class_label"][idx]  
        
        #pull out the data
        packet = self._wave_data[data_idx][time_ind:(time_ind + self.pkt_size)]

        return packet, class_label
            
    def get_class_weights(self):
        class_labels = self._packet_table['class_label']
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(class_labels),
                                             y=class_labels)
        return torch.tensor(class_weights, dtype=torch.float32)
    
    
    
class ICDataTrials(Dataset):
    """
    Main class for raw datasets for autoencoder training
    files are stored in a pandas dataframe called _data_table
    each file is an experiment from one channel of ieeg data
    
    files are loaded then decomposed into packets of 500ms
    then normalized to zero mean, unit variance, 
    then bounded by the maximum value
    
    do data augmentation is performed
    """
    def __init__(self, trialinfo, data_root, training,
                 pkt_size = 512, step_size = 50):
        """
        initialization for the autoencoder raw data class

        Parameters
        ----------
        training_file : path
            location of csv file holding the filenames for training and testing
        distribution_file : path
            filename of the tdigest distribution dict that gives percentiles
            of the descriptor data
        training : bool, optional
            focus on the training or testing rows?. The default is True.
        nsamps : int, optional
            number of random windows to extract from each datafile.
            The default is 100.
        pkt_size : int, optional
            length of the datapackets. The default is 500.

        Returns
        -------
        None.

        """
        
        #load the data table that contains all the file names
        self.training = training
        self._data_table = trialinfo[
            trialinfo.training == training].reset_index(drop=True)
        self.data_root = data_root / 'trial'
        self.pkt_size = pkt_size
        self.step_size = step_size
        
        #load all the subject's data
        self._load_data()
        
        #prepare generators
        if training:
            self.seed = int(time.time())
        else:
            self.seed = int(42)
        self.generator = torch.Generator()
        self.reset_seed()
            
    def __len__(self):
        """
        length of the datset:
            determined by the number of rows in the internal dataframe

        Returns
        -------
        int
            length of the dataframe.

        """
        return len(self._packet_table)
        
    
    def _load_data(self):
        
        data_table = self._data_table
        wave_data = []
        paket_rows = []
        #loop through and aload all files
        for index, trial_row in data_table.iterrows():
            trial_ind = trial_row['relative_trial']
            
            # Load the trial_dict from the pickle file
            trial_file = self.data_root / f'trial_{trial_ind}.pkl'
            with open(trial_file, 'rb') as file:
                trial_dict = pickle.load(file)
            
            #pull out the data
            wave = trial_dict['wave']
            wave_data.append(wave)
            
            #make the packet table for this trial
            wave_len = wave.shape[1]
            pkt_size = self.pkt_size
            step_size = self.step_size
            num_windows = (wave_len - pkt_size) // step_size
            wave_start = np.arange(num_windows)*step_size
            remainder = wave_len - (wave_start[-1] + pkt_size + step_size)
            if remainder > 2:
                start_add = int(remainder//2)
            else:
                start_add = 0
            
            trial_row['trial_ind'] = index
            repeated_rows = pd.DataFrame([trial_row] * num_windows)
            repeated_rows['start_ind'] = wave_start + start_add
            paket_rows.append(repeated_rows)
            
        #add data to self
        self._packet_table = pd.concat(paket_rows, ignore_index=True)
        self._wave_data = wave_data
            
    def __getitem__(self, idx):
        
        
        #pull out relevant information from dataframe
        data_idx = self._packet_table["trial_ind"][idx]
        start_ind = self._packet_table["start_ind"][idx]
        class_label = self._packet_table["class_label"][idx]  
        '''
        #pul1l out the data
        packet = self._wave_data[data_idx][time_ind]

        return packet, class_label
        '''
        
        #random start for training data
        if self.training:
            #get the starting index
            _rand = torch.rand(1, generator=self.generator).item()
            max_start = self.step_size - 1
            rand_start = int(_rand * max_start)
        else:
            rand_start = 0
            
        #pull out the packets
        packet = self._wave_data[data_idx][:,(rand_start + start_ind):
                             (rand_start + (start_ind + self.pkt_size))]
        
        #normalize voltage distributions
        one_wave = self.normalize(packet)
        
        return one_wave, class_label
            
            
    def normalize(self, packet):
        
        #calc meta features
        mean = np.mean(packet)
        std = np.std(packet)
        
        #normalize packet
        packet_norm = (packet - mean)/(std + np.finfo(float).eps)
        
        #find and clip at 99th percentile
        max_clip = np.percentile(np.abs(packet_norm),98.4)
        packet_norm = np.clip(packet_norm,
                              a_min=-1*max_clip,
                              a_max=max_clip)
        #set maximum value to 1
        packet_norm = packet_norm/(
            max_clip + np.finfo(float).eps)
        packet_norm = torch.tensor(packet_norm)
        
        return packet_norm

    def reset_seed(self):
        self.generator.manual_seed(self.seed)
        
    def shuffle_packet_table(self):
        '''
        function shuffles observations in the packet table
        keeping the structure of files together in order

        Returns
        -------
        None.

        '''
        # Step 1: Split the table into tables that have the same number in the data_idx column
        grouped_tables = [group for _, group in 
                          self._packet_table.groupby('data_idx')]
        
        # Step 2: Shuffle the rows within each of these tables
        shuffled_tables = [table.sample(frac=1).reset_index(drop=True)
                           for table in grouped_tables]
        
        # Step 3: Shuffle the order of these tables and stack them
        np.random.shuffle(shuffled_tables)
        self._packet_table = pd.concat(shuffled_tables).reset_index(drop=True)
    
    def get_class_weights(self):
        class_labels = self._packet_table['class_label']
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(class_labels),
                                             y=class_labels)
        return torch.tensor(class_weights, dtype=torch.float32)
        
        
        
def  get_sbj_table_pretrain(train_table, sbj_name, data_root):
    '''
    function for getting subject specific table from master table
    used for pretraining based on subject specific training and testing

    Parameters
    ----------
    train_table : table containing subject information
    sbj_name : str
        subject name.
    data_root : Path
        path to data

    Returns
    -------
    valid_rows : data frame for subject specific data table
    num_chan : number of channels for this subject
    weights : classification weights to tune class loss appropriately

    '''
    #get subject specific table
    train_table = train_table[
        train_table['subject'] == sbj_name
        ].reset_index(drop=True)
    
    #get number of channels in each
    sbj_table = pd.DataFrame()
    unique_projects = train_table['project'].unique()
    for project in unique_projects:
        # Filter the rows for the current project
        project_rows = train_table[train_table['project'] == project]
        unique_blocks = project_rows['block_num'].unique()
        for block in unique_blocks:
            block_rows = project_rows[project_rows['block_num'] == block]
            block_len = len(block_rows)
            block_row = block_rows[block_rows['chan'] == 0].reset_index(drop=True)
            block_row['chan'] = block_len
            sbj_table = pd.concat([sbj_table, block_row])
            
    sbj_chan_dict = {
                'S23_199_GB':165,
                'S23_205_LLC':170,
                'S23_206_SO':112,
                'S22_177_JM':154,
                'S21_169_BH':169,
                'S22_176_LB':143,
                'S23_211_SS':183, 
                'S22_183_CR':166, 
                'S22_178_AF':203, 
                'S23_207_SO':105,
                'S23_212_JM':199, 
                'S21_166_TM':108,
                 }
    
    num_chan = sbj_chan_dict[sbj_name]
    sbj_table = sbj_table[sbj_table['chan'] >= num_chan].reset_index(drop=True)
    
    #classify project data
    # 1. Find all unique entries in the column "project"
    unique_projects = sbj_table['project'].unique()

    # Initialize an empty DataFrame to store the valid rows
    valid_rows = pd.DataFrame()

    # 2. Loop through each unique project
    for project in unique_projects:
        # Filter the rows for the current project
        project_rows = sbj_table[sbj_table['project'] == project]

        # Check if the project has at least one row with training == True and one with training == False
        if (project_rows['training'] == True).any() and (project_rows['training'] == False).any():
            # Append the valid rows to the new DataFrame
            valid_rows = pd.concat([valid_rows, project_rows])

    # 3. Create a new table with a new index that only consists of these valid rows
    valid_rows.reset_index(drop=True, inplace=True)
    
    # Create a unique integer label for each project
    unique_projects = valid_rows['project'].unique()
    project_to_label = {project: i for i, project in enumerate(unique_projects)}
    
    # Replace the "class_label" column with these unique labels
    valid_rows['class_label'] = valid_rows['project'].map(project_to_label)
    
    # Calculate the number of observations for each class where "training" == True
    num_observations = valid_rows[valid_rows['training'] == True].groupby('class_label')['num_windows'].sum()
    
    # Total number of observations
    total_observations = num_observations.sum()

    # Calculate the weights for categorical cross-entropy
    weights = total_observations / (num_observations * len(num_observations))
    
    return valid_rows, num_chan, weights


def get_sbj_table_JEPA(train_table, sbj_name, data_root):
    
    #get subject specific table
    train_table = train_table[
        train_table['subject'] == sbj_name
        ].reset_index(drop=True)
    
    #get number of channels in each
    sbj_table = pd.DataFrame()
    unique_projects = train_table['project'].unique()
    for project in unique_projects:
        # Filter the rows for the current project
        project_rows = train_table[train_table['project'] == project]
        unique_blocks = project_rows['block_num'].unique()
        for block in unique_blocks:
            block_rows = project_rows[project_rows['block_num'] == block]
            block_len = len(block_rows)
            block_row = block_rows[block_rows['chan'] == 0].reset_index(drop=True)
            block_row['chan'] = block_len
            sbj_table = pd.concat([sbj_table, block_row])
            
    sbj_chan_dict = {
                'S23_199_GB':165,
                'S23_205_LLC':170,
                'S23_206_SO':112,
                'S22_177_JM':154,
                'S21_169_BH':169,
                'S22_176_LB':143,
                'S23_211_SS':183, 
                'S22_183_CR':166, 
                'S22_178_AF':203, 
                'S23_207_SO':105,
                'S23_212_JM':199, 
                'S21_166_TM':108,
                 }
    
    num_chan = sbj_chan_dict[sbj_name]
    sbj_table = sbj_table[sbj_table['chan'] >= num_chan].reset_index(drop=True)
    
    return sbj_table, num_chan


def get_min_chan_from_sbj_table(data_root, sbj_table):
    '''
    function counts the number of matfiles for each subject excluding
    additional channels added during stay

    Parameters
    ----------
    data_root : Path
        path to data.
    sbj_table : pandas dataframe
        dataframe holding subject data.

    Returns
    -------
    min_chan: minimum chans across blocks

    '''
    min_chan = 10000
    #loop through rows and blocks check the number of files in each location
    for _, row in sbj_table.iterrows():
        #get the 
        proj_path = data_root / row['subject'] / row['project'] 
        block_path = proj_path / f'block_{row["block_num"]}'
        #count the mat files
        mat_files = list(block_path.glob('*.mat'))
        num_chan = len(mat_files)
        min_chan = np.min([min_chan,num_chan])
        
    return min_chan