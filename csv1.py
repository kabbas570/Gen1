import pandas as pd
csv_path=r'C:\My_Data\M2M Data\dataset_information.csv'
df = pd.read_csv(csv_path)

# vendors = df['VENDOR']
# scanners = df['SCANNER']
# diseases=df['DISEASE']
# fields=df['FIELD']

#u_items = set(fields)

images_folder=r'C:\My_Data\M2M Data\data'


import timm 
import torch
import torch.nn as nn
import math
from torchinfo import summary
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision import transforms
import pandas as pd

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from torch.utils.data import Dataset
           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True

class CustomDataset(Dataset):
    
    def __init__(self, df, images_folder):
        #df = pd.read_csv(csv_path)
        self.df = df
        self.images_folder = images_folder
        
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']
        
        
        self.images_name = df['SUBJECT_CODE'] 


    def __len__(self):
        return self.vendors.shape[0]

    def __getitem__(self, index):
        #img_path=os.path.join(self.images_folder,
                              #self.images_name[index])
        print(index)
        #image=Image.open(img_path)
        #image=cv2.imread(img_path)
        # c1=self.df1.loc[self.df1['Long Slide ID']==self.images_name[index]]
        # #print(c1)
        # c1 = c1.drop(columns=['Grade','Long Slide ID','Gender','age']) # this for genss
        # cLin_features = np.array(c1)
        # #print(c1.shape)
        
        
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        fields_ = self.fields[index]
        
        

        
        # y_label = self.y[index]
        
        
        # if self.transform is not None:
        #     #image = self.transform(image)
        # return (image,cLin_features,y_label,self.images_name[index])
        
        # print(self.images_name[index])
        
        return scanners_,vendors_,diseases_,fields_,self.images_name[index]
        
        

def Data_Loader(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = CustomDataset(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

batch_size=10
train_loader=Data_Loader(df,images_folder,batch_size)

print(len(train_loader))


a=iter(train_loader)
a1=next(a)

scanner=a1[0]
vendor=a1[1]
disease=a1[2]
fields=a1[3]
ID=a1[4]







