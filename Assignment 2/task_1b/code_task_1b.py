
"""
Created on Wed April 13 17:43:13 2020

@author: Sivaraman SIvaraj
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
input_data = np.load('x1.npy')
input_data = input_data.astype('float32') # will give double type tensor
output_class = np.load('y1.npy')
x_all = torch.from_numpy(input_data)# batch size , 1658
y = torch.from_numpy(output_class)
y_all = y.type(torch.LongTensor) # for label, dtype should be long
x_all = (x_all - x_all.min(0)[0]) / (x_all.max(0)[0] - x_all.min(0)[0]) #Normalising data
x_pt = x_all[:1326] # data for pretraining
y_pt = y_all[:1326]
x_val = x_all[1326:] # data for finetuning
y_val = y_all[1326:]


class AANN(nn.Module):
    def __init__(self, D_in,L1,H,L3):
         #D_in - input dimension, L1 - layer1 dimension, H - Bottleneck layer dimension
         #L3 - layer3 dimension
         super(AANN, self).__init__()
        
         self.fc1 = nn.Linear(D_in, L1)
         self.fc2 = nn.Linear(L1, H)
         self.fc3 = nn.Linear(H, L3)
         self.fc4 = nn.Linear(L3, D_in) #D_in and D_out is same for AANN

    def forward(self, x):

         x = torch.sigmoid(self.fc1(x))
         bl = self.fc2(x) # bl - bottleneck layer
         y = torch.sigmoid(self.fc3(bl))
         y = self.fc4(y)
         return y,bl
     
def PreTraining(x,lr,n_epoch,net):
# x - input, lr - learning rate, n_epoch - number of epochs, net - Neuaral Network instance
     N = len(x)#batch size
     criterion = nn.MSELoss()#mean square error
     optimizer = optim.Adam(net.parameters(), lr,betas = (0.9,0.999))#gradientupdate
     mini_batch = 10
     lossM = []
     for epoch in range(n_epoch):
         e = 0
         print(epoch)
         for i in range(0,N-mini_batch+1,mini_batch):
             input = x[i:i+mini_batch]
             target = input
             optimizer.zero_grad() # zero the gradient buffers
             output = net(input)[0]
             loss = criterion(output, target)
             loss.backward()
             optimizer.step() # Does the update
             e += loss.item()
         lossM.append(e*mini_batch/N)
     with torch.no_grad():
        bottle_neck_layer = net(x)[1] #get the bottle neck layer nodes output
     return net.fc1.weight.data,net.fc1.bias.data,net.fc2.weight.data,net.fc2.bias.data,bottle_neck_layer,lossM


#w1 - weight matrix of fc1, b1 - bias of fc1, w2 - weight matrix of fc2, b2 - bias of fc2
#bottleneck = bottle neck layer 1 output, e - AANN training error
"Auto Encoder Stack 1"
AANN1 = AANN(828,414,414,414)
w1_1,b1_1,w2_1,b2_1,bottleneck1,e1 = PreTraining(x_pt,0.001,120,AANN1)
plt.plot(e1)
"Auto Encoder Stack 2"
AANN2 = AANN(414,207,207,207)
w1_2,b1_2,w2_2,b2_2,bottleneck2,e2 = PreTraining(bottleneck1,0.001,20,AANN2)
plt.plot(e2)
"Auto Encoder Stack 3"
AANN3 = AANN(207,100,100,100)
w1_3,b1_3,w2_3,b2_3,_,e3 = PreTraining(bottleneck2,0.001,40,AANN3)
plt.plot(e3)
"Merging the Auto Encoder weights"
w1,b1 = w1_1,b1_1
w4,b4 = w2_3,b2_3
w2 = torch.mm(w1_2,w2_1)
b2 = b1_2 + torch.mm(b2_1.view(1,-1),w1_2.t())
w3 = torch.mm(w1_3,w2_2)
b3 = b1_3 + torch.mm(b2_2.view(1,-1),w1_3.t())

"Classifier network"
class Classifier_Net(nn.Module):
    def __init__(self):
         super(Classifier_Net, self).__init__()
        
         self.fc1 = nn.Linear(828,414) # has pre trained weight
         self.fc2 = nn.Linear(414,207) # has pre trained weight
         self.fc3 = nn.Linear(207,100) # has pre trained weight
         self.fc5 = nn.Linear(100,100) # has pre trained weight
         self.fc4 = nn.Linear(100,5) # output layer
         
    def forward(self, x):
         x = torch.sigmoid(self.fc1(x))
         x = torch.sigmoid(self.fc2(x))
         x = torch.sigmoid(self.fc3(x))
         x = self.fc5(x)
         x = torch.softmax(self.fc4(x),dim=1)
         return x
     
net = Classifier_Net() #instance of classifier network
"Confusion matrix calculation"
def confusion_matrix(x,t):
    mat = torch.zeros(5,5)
    with torch.no_grad():
        op = net(x)
    y = torch.argmax(op,dim=1)
    for i in range(5):
        for j in range(5):
             c = (t == i)
             d = (y == j)
             mat[i,j] = torch.sum(c & d).type(torch.IntTensor)
    acc = torch.sum(torch.diag(mat)) / torch.sum(mat)
    return mat,acc

"Initialising weights with pretrained weights"
with torch.no_grad():
     net.fc1.weight.data = w1
     net.fc1.bias.data = b1
     net.fc2.weight.data = w2
     net.fc2.bias.data = b2
     net.fc3.weight.data = w3
     net.fc3.bias.data = b3
     net.fc5.weight.data = w4
     net.fc5.bias.data = b4
     
# Training the SAE based classifier
x_ft,y_ft = x_pt[:165],y_pt[:165] # data for finetuning
loss_train =[]
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001,betas = (0.9,0.999))
N = len(y_ft)
batch_size = 10


for epoch in range(20): # loop over the dataset multiple times
     running_loss = 0.0
     print(epoch)
     for i in range(0,N-batch_size+1,batch_size):
         input = x_ft[i:i+batch_size,:]
         label = y_ft[i:i+batch_size]
         # zero the parameter gradients
         optimizer.zero_grad()
        
         # forward + backward + optimize
         output = net(input)
         loss = criterion(output, label)
         loss.backward()
         optimizer.step() # does the update
        
         running_loss += (loss.item()*batch_size)/N # average of all losses
     loss_train.append(running_loss)
plt.plot(loss_train)

conf_val, acc_val = confusion_matrix(x_val,y_val) # results for validation dataset
conf_pt, acc_pt = confusion_matrix(x_pt,y_pt) # results for pretraining dataset
conf_ft, acc_ft = confusion_matrix(x_ft,y_ft) # results for finetuning dataset
print(acc_val,acc_pt,acc_ft)









