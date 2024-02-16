from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 0) Load Dataset
bc = datasets.load_iris()
x,y = bc.data, bc.target
n_samples, n_features  = x.shape
out_features = len(set(y))

# creating train-test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Scale data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Converting numpy array to tensors
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# 1) Model
class NeuralNet(nn.Module):
    def __init__(self,in_features, h1, h2, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

model = NeuralNet(in_features=n_features,h1=50,h2=50,out_features=out_features).to(device)

# 2) Loss and Optimizer
lr = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# 3) Training loop
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    model.train()
    y_pred = model(x_train)         # Forward pass

    loss = loss_fn(y_pred,y_train)  # calculate loss
    losses.append(loss.detach().numpy)
    # Calculating accuracy
    _, y_pred_cor = torch.max(y_pred,1)
    correct_pred = (y_pred_cor == y_train).sum().item()
    acc = correct_pred / (len(y_train)) 
    
    loss.backward()                 # backpropogate losses
    optimizer.step()                # update gradients
    optimizer.zero_grad()           # clear gradient for next forward pass
    
    if (epoch +1) % 100 == 0:
        print(f'epoch : {epoch +1} | loss : {loss.item():.4f} | accuracy: {acc:.4f}')
        

model.eval()
with torch.inference_mode():
    y_pred = model(x_test)
    loss = loss_fn(y_pred, y_test)
    # Calculating accuracy
    _, y_pred_cor = torch.max(y_pred,1)
    correct_pred = (y_pred_cor == y_test).sum().item()
    acc = correct_pred / (len(y_test)) 
    print(f'Model evaluation results:\nLoss : {loss.item():.4f} | accuracy : {acc:.4f}')