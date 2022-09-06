import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#####################################################
################### Data Creation ###################
#####################################################
def f(N):
    X,Y  = list(), list()
    for i in range(N):
        x_temp = np.random.randint(-100,100)
        y_temp = np.random.randint(-100,100)
        if x_temp == 0:
            x_temp  = 1
        if y_temp == 0:
            y_temp = -1
        X.append([float(x_temp),float(y_temp)]) 
        ####################################
        if x_temp > 0 and y_temp > 0:
            Y.append([1.0,0.0,0.0,0.0])
        elif x_temp < 0 and y_temp > 0:
            Y.append([0.0,1.0,0.0,0.0])
        elif x_temp < 0 and y_temp < 0:
            Y.append([0.0,0.0,1.0,0.0])
        else:
            Y.append([0.0,0.0,0.0,1.0])
    return X, Y

Xl,Yl = f(2000)
#######################################################
##################### MLFNN ###########################
#######################################################


class MLFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLFFNN, self).__init__()
        self.ip  = nn.Linear(input_size, hidden_size)
        self.HL1 = nn.Linear(hidden_size, hidden_size)
        self.HL2 = nn.Linear(hidden_size, hidden_size)
        self.op  = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        x = torch.sigmoid(self.ip(state))
        x = torch.sigmoid(self.HL1(x))
        x = torch.sigmoid(self.HL2(x))
        x = torch.softmax(self.op(x),dim=-1)
        return x

net             = MLFFNN(2,128,4)
criterion       = nn.CrossEntropyLoss()
optimizer       = torch.optim.Adam(net.parameters(),lr=1e-3) 
# ss = torch.tensor([5.0,2.0])
# print(net(ss))
#########################################################
############## Training the Neural Network ##############
#########################################################
X,Y    = torch.tensor(Xl),torch.tensor(Yl)
def Train_FFNN(No_Episodes):
    CrossEL        = list()
    for i in range(No_Episodes):
        if (i%50) == 0:
            print("Episode : ", i)
        yp      = net(X)
        loss    = criterion(yp,Y)
        CrossEL.append(loss.item())
        ###########################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), "W.pt")  
    return CrossEL
        
CEL = Train_FFNN(2000)    
np.save("CEL.npy",CEL)
#########################################################
################### Results Plotting ####################
#########################################################

plt.figure(figsize=(9,6))
plt.plot(CEL,color="crimson",label="Cross Entropy Error")
plt.title("Multilayer Feed Forward Nueral Network - Classification Training")
plt.xlabel("No of Epochs")
plt.ylabel("Cross Entropy Error")
plt.grid()
plt.legend(loc = "best")
# plt.axhline(y=0,color="k")
# plt.axvline(x=0, color="k") 
plt.savefig("MSE.jpg",dpi = 420)
plt.show()

#########################################################
#########################################################


    
    
    
    
    
    
    
    
    
    
    