import torch
from torch.utils.data import Dataset, random_split
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy,self).__init__()

        data = np.load(np_dataset)
        self.db_S = data['db_S'] 
        self.db_data = data['db_data']          
        self.db_label = data['db_label'].astype(float)  
        self.N = self.db_S.shape[-1]
        assert self.db_S.shape[-1] == self.N
        assert self.db_data.shape[-1] == self.N
        assert self.db_label.shape[-1] == self.N 
        
        self.db_label = self.db_label[0:2,:]
        self.preprocess_label()
        
        self.db_S = torch.from_numpy(self.db_S).permute(3, 2, 0, 1).float()
        self.db_data = torch.from_numpy(self.db_data).permute(2, 1, 0).float().unsqueeze(3)
        self.db_label = torch.from_numpy(self.db_label).permute(1,0).float()
        #self.__getitem__([1,2])


    def preprocess_label(self):
        sbp = self.db_label[0,:]
        dbp = self.db_label[1,:]
        #hr = self.db_label[2,:]
        self.sbp_max = np.max(sbp)
        self.sbp_min = np.min(sbp)
        self.dbp_max = np.max(dbp)
        self.dbp_min = np.min(dbp)
        #self.hr_max = np.max(hr)
        #self.hr_min = np.min(hr)
        sbp = (sbp - self.sbp_min)/(self.sbp_max - self.sbp_min)
        dbp = (dbp - self.dbp_min)/(self.dbp_max - self.dbp_min)
        #hr = (hr - self.hr_min)/(self.hr_max-self.hr_min)
        self.db_label[0,:] = sbp 
        self.db_label[1,:] = dbp 
        #self.db_label[2,:] = hr
    
    def get_label_scale(self):
        #return [self.sbp_max,self.sbp_min ,self.dbp_max, self.dbp_min,self.hr_max, self.hr_min ]
        return torch.from_numpy(np.array([self.sbp_max,self.sbp_min,self.dbp_max, self.dbp_min])).float()
        

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        S_i_ads = self.db_S[idx,0:1,...]
        S_i_afe = self.db_S[idx,1:2,...]
        data_i_ads = self.db_data[idx,0:1,...]
        data_i_afe = self.db_data[idx,1:2,...]
        label_i = self.db_label[idx,0:2]

        return data_i_ads,data_i_afe,S_i_ads, S_i_afe,label_i
    
    
def generate_train_subject(data_files):
    dataset = LoadDataset_from_numpy(data_files)
    test_partial = 0.8 
    train_size = int(test_partial * len(dataset))
    val_size = len(dataset) - train_size
    lengths = [train_size, val_size]
    train_dataset, test_dataset = random_split(dataset, lengths, generator=generator)
    return train_dataset, test_dataset, dataset.get_label_scale()

def data_generator_np(data_files, batch_size):
    train_dataset, test_dataset,label_scale = generate_train_subject(data_files)
    # train_dataset = LoadDataset_from_numpy(training_files)
    # test_dataset = LoadDataset_from_numpy(subject_files)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, label_scale
