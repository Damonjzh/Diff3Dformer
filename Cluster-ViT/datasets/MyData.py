'''
Dataset for training
Written by Whalechen
'''

from os import listdir, mkdir
import random
import numpy as np
from torch.utils.data import Dataset
import random
import torch
class MyDataset(Dataset):

    def __init__(self, root_dir,sequence_len,max_num_cluster,status='train',input_pool=False, data_index = None):
        # self.patientData_list = [root_dir + "/"+ pth for pth in listdir(root_dir)]
        if data_index is None:
            self.patientData_list = [root_dir + "/"+ pth for pth in listdir(root_dir)]
        else:

            # fld
            # self.patientData_list = [root_dir + "/" + pth for pth in listdir(root_dir) for index in data_index if
            #                          int(pth.split('_')[0]) == index]
            #ccccii
            self.patientData_list = [root_dir + "/" + pth for pth in listdir(root_dir) for index in data_index if
                                     pth.split('.')[0] == index]

        self.sequence_len=sequence_len
        self.max_num_cluster=max_num_cluster
        self.status = status
        self.input_pool = input_pool
    def __len__(self):
        return len(self.patientData_list)

    def __getitem__(self, idx):

        temp = np.load(self.patientData_list[idx],allow_pickle=True).item()
        patientEmbedding=temp['patientEmbedding']
        pos=temp['position']
        cluster=temp['cluster']

        Dead=temp['Dead']
        followUpTime=temp['FollowUpTime']

        # balance data
        # if self.status == 'train':
        #     group_size = 26
        #
        #     if len(patientEmbedding) % group_size == 0:
        #         num_groups = len(patientEmbedding) // group_size
        #     else:
        #         num_groups = len(patientEmbedding) // group_size + 1
        #
        #     selected_indices = []
        #     for i in range(num_groups):
        #         start_index = i * group_size
        #         end_index = (i + 1) * group_size
        #         if i == num_groups - 1:
        #             end_index = len(patientEmbedding)
        #         try:
        #             selected_index = random.randint(start_index, end_index - 1)
        #         except:
        #             print(start_index, end_index)
        #             print(len(patientEmbedding))
        #
        #         selected_indices.append(selected_index)
        #     patientEmbedding = patientEmbedding[selected_indices]
        #     pos = pos[selected_indices]
        #     cluster = cluster[selected_indices]


        if not self.input_pool:
            # No pooling 
            patientEmbedding = torch.tensor(patientEmbedding)
            pos = torch.tensor(pos)
            patientEmbedding = torch.cat((patientEmbedding,torch.zeros(self.sequence_len-patientEmbedding.shape[0],patientEmbedding.shape[1])))

            pos = torch.cat((pos,torch.zeros(self.sequence_len-pos.shape[0],pos.shape[1])))
            keyPaddingMask = torch.cat((torch.zeros(cluster.shape[0]),torch.ones(self.sequence_len-cluster.shape[0])))
            keyPaddingMask = keyPaddingMask.type(torch.ByteTensor)
            cluster = torch.tensor(cluster).to(torch.int64).squeeze()
            cluster = torch.cat((cluster,self.max_num_cluster*torch.ones(self.sequence_len-cluster.shape[0]))).to(torch.int64)
            Dead = torch.tensor(Dead).to(torch.int64)
            followUpTime = torch.tensor(followUpTime).to(torch.float32)
            try:
                patient_name = torch.tensor(int(self.patientData_list[idx].split('/')[-1].split('.')[0])).to(torch.int64)
            except:
                patient_name = torch.tensor(int(self.patientData_list[idx].split('/')[-1].split('_')[0])).to(torch.int64)
            # patchidx = torch.tensor(patchidx).to(torch.int64)
            # data processing
        else:           
            selectedIndex = random.choices(range(len(patientEmbedding)),k=self.sequence_len)
            patientEmbedding = torch.tensor(patientEmbedding[selectedIndex,:])
            pos = torch.tensor(pos[selectedIndex,:])
            keyPaddingMask = torch.zeros(self.sequence_len).type(torch.ByteTensor)
            cluster = cluster[selectedIndex,:]
            cluster = torch.tensor(cluster).to(torch.int64).squeeze()
            Dead = torch.tensor(Dead).to(torch.int64)
            followUpTime = torch.tensor(followUpTime).to(torch.float32) 

        # return (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patient_name,patchidx)
        return (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patient_name)