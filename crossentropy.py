import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual,predicted):
    loss = -np.sum(actual * np.log(predicted)) / len(predicted)
    return loss   # / float(predicted.shape[0])

# y must be one one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
y = np.array([1,0,0])


# y_pred has probabilites
y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(y,y_pred_good)
l2 = cross_entropy(y,y_pred_bad)
print(f'Loss 1 numpy : {l1:.4f}')
print(f'Loss 2 numpy : {l2:.4f}')

# In pytorch crossentropyloss applies softmax beforehand and therefore is not required
# y (true) has class labels, not one-hot encoding
# y_pred has raw scores (logits), no softmax
loss_fn = torch.nn.CrossEntropyLoss()
y = torch.tensor([0])
# nsamples x nclasses = 1x3
y_pred_good = torch.tensor([[2.0,1.0,0.1]])
y_pred_bad = torch.tensor([[0.5,2.0,0.3]])

l1 = loss_fn(y_pred_good,y)
l2 = loss_fn(y_pred_bad,y)

_, predications1 = torch.max(y_pred_good,1)
_, predications2 = torch.max(y_pred_bad,1)

print(f'Loss 1 torch : {l1.item()} with position : {predications1}')
print(f'Loss 2 torch : {l2.item()} with position : {predications2}')

