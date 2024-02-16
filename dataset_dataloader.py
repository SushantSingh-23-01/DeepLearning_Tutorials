# Epoch = One forward and backward pass of ALL training samples
# batch_size =  number of training samples in forward and backward pass
# number of iterations =  number of passes, each pass using (batch_size) number of samples
# e.g for 100 samples with batch size 20 --> number of iteratiosn  = 100/20 = 5

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class wineDataset(Dataset):
    def __init__(self):
        # Data Loading
        # https://github.com/patrickloeber/pytorchTutorial/blob/master/data/wine/wine.csv
        xy = np.loadtxt(r'dataset/wine.csv',delimiter= ',',dtype=np.float32,skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
    
    def __getitem__(self,index):
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
        
dataset = wineDataset()

dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)

dataiter = iter(dataloader)
data = next(dataiter)
features,labels = data

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(f'total samples: {total_samples} and number of iterations {n_iterations}')

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward pass and optimize
        if (i+1)% 5 == 0:
            print(f'epoch : {epoch+1} / {num_epochs} | step : {i+1}/{n_iterations} | inputs : {inputs.shape}')