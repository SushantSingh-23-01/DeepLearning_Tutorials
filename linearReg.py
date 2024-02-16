import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets


# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples = 100,
                                            n_features=1,
                                            noise=20,
                                            random_state=1
                                            )
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
lr = 1e-2
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# 3) Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(X)           # Forward Pass     
    loss = loss_fn(y,y_pred)    # compute loss
    loss.backward()             # backpropogate gradients
    optimizer.step()            # sum gradinent and update weight
    optimizer.zero_grad()       # clear optimizer gradients
    
    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}  |  loss: {loss.item():.4f}')