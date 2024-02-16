from sklearn.discriminant_analysis import StandardScaler
import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare dataset
bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=1234)

# scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test= y_test.view(y_test.shape[0],1)

# 1) Model
class LogisiticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisiticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisiticRegression(n_features)

# 2) Loss and Optimizer
lr = 1e-3
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# 3) training loop
num_epoch = 100
for epoch in range(num_epoch):
    y_pred = model(x_train)             # Forward Pass
    loss = loss_fn(y_pred,y_train)      # computing loss
    loss.backward()                     # backward pass gradients
    optimizer.step()                    # update gradients
    optimizer.zero_grad()               # clear gradients
    
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1} | loss = {loss.item():.4f}')

# 4) Evaluate
with torch.no_grad():
    y_pred = model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum()  / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')