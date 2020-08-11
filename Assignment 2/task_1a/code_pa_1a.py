"""
Created on Tue April 11 11:12:26 2020

@author: Sivaraman Sivaraj
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
X = np.load('x1.npy')
X_train = X[0:1161,:] #70% for training
X_val = X[1161:,:] #30% for validation
mu = np.mean(X_train,axis=0).reshape(1,828)
sig = np.std(X_train,axis=0).reshape(1,828)
X_train = (X_train - mu) / sig
X_val = (X_val - mu) / sig
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
x_train = torch.from_numpy(X_train)
x_val = torch.from_numpy(X_val)
"nodes and hidden layers specification"
N = len(X_train) # batch size
D_in = len(X_train[0])# input dimension 828
M = D_in//2
B = 350 # Bottle neck layer dimension
D_out = D_in # same as input, because auto associative neural network
"Defining AANN for feature extraction"
class Net(nn.Module):
    def __init__(self):
         super(Net, self).__init__()
        
         self.fc1 = nn.Linear(D_in, M)
         self.fc2 = nn.Linear(M, B)
         self.fc3 = nn.Linear(B, M)
         self.fc4 = nn.Linear(M, D_out)
    
    def forward(self, x):
    
         x = torch.tanh(self.fc1(x))
         h = self.fc2(x) # bottle neck layer
         y = torch.tanh(self.fc3(h))
         y = self.fc4(y)
         return y,h
net = Net()
lossM = []
lossval = []
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.0001,betas=[0.9,0.999])

mini_batch = 10
for epoch in range(15):
     tot_loss = 0
     print(epoch)
     for i in range(0,N-mini_batch+1,mini_batch):
         input = x_train[i:i+mini_batch]
         target = input
         optimizer.zero_grad() # zero the gradient buffers
         output = net(input)[0]
         loss = criterion(output, target)
         loss.backward()
         optimizer.step()
         tot_loss += loss.item()
         lossM.append(tot_loss*mini_batch/N)
plt.plot(lossM)
plt.plot(lossval)

with torch.no_grad():
     bl_train = net(x_train)[1] #bottle neck layer train
     bl_val = net(x_val)[1] #bottle neck layer validation
     
def calc_confmat(targ,pred,num_classes): #function to calculate confusion matrix
    # targ - target class
    # pred - predicted class
    # num_classes - numbers of classes
     conf = torch.zeros(num_classes,num_classes)
     for i in range(num_classes):
         for j in range(num_classes):
             c = (targ == i)
             d = (pred == j)
             conf[i,j] = torch.sum(c & d).type(torch.IntTensor)
     return conf
output_class = np.load('y1.npy')
y = torch.from_numpy(output_class)
y = y.type(torch.LongTensor)
y_train = y[0:1161]
y_val = y[1161:]
D_in = len(bl_train[0]) #dimension of reduced features
H1 = int(np.ceil(D_in/2))
H2 = int(np.ceil(H1/2))
D_out = np.size(np.unique(y))

class Classifier(nn.Module):
    def __init__(self):
         super(Classifier, self).__init__()
        
         self.fc1 = nn.Linear(D_in, H1) 
         self.fc2 = nn.Linear(H1, H2)
         self.fc3 = nn.Linear(H2, D_out)
         
    def forward(self, x):

         x = torch.relu(self.fc1(x))
         x = torch.relu(self.fc2(x))
         x = torch.softmax(self.fc3(x),dim=1)
         return x
     
classifier = Classifier()
lossT = []
N = len(X_train)
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(classifier.parameters(), lr=0.001, betas=[0.9,0.999])


for epoch in range(10): # epochs
     print(epoch)
     tot_loss = 0
     mini_batch = 10
     for i in range(0,N-mini_batch+1,mini_batch):
         input = bl_train[i:i+mini_batch]
         target = y_train[i:i+mini_batch]
         optimizer1.zero_grad()
         output = classifier(input)
         loss = criterion1(output, target)
         loss.backward()
         optimizer1.step()
         tot_loss += loss.item()
     lossT.append(tot_loss/N)
plt.plot(lossT)

with torch.no_grad():
     out = classifier(bl_train)
     outval = classifier(bl_val)
     
_, predicted = torch.max(out,1)
_, predval = torch.max(outval,1)
conf_train = calc_confmat(y_train,predicted,D_out)
acc_train = torch.sum(torch.diag(conf_train)) / torch.sum(conf_train)
conf_val = calc_confmat(y_val,predval,D_out)
acc_val = torch.sum(torch.diag(conf_val)) / torch.sum(conf_val)
print(acc_train,acc_val)








