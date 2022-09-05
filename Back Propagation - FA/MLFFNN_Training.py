import numpy as np
import matplotlib.pyplot as plt
#####################################################
################### Data Creation ###################
#####################################################
def f():
    X  = np.arange(-100,100,0.01)
    X  = X.tolist()
    Y  = list()
    ##############################
    for i in range(len(X)):
        temp = 3*(X[i])**2
        Y.append([temp])
    ##############################
    X = np.array(X).reshape(len(X),1)
    X  = X.tolist()
    return X, Y

Xw,Yw = f()
Xl,Yl = Xw[::2],Yw[::2]
Xt,Yt = Xw[1::2],Yw[1::2]
#######################################################
##################### MLFNN ###########################
#######################################################
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd

class MLFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLFFNN, self).__init__()
        self.ip  = nn.Linear(input_size, hidden_size)
        self.HL1 = nn.Linear(hidden_size, hidden_size)
        self.HL2 = nn.Linear(hidden_size, hidden_size)
        self.op  = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        x = F.relu(self.ip(state))
        x = F.relu(self.HL1(x))
        x = F.relu(self.HL2(x))
        x = self.op(x)
        return x

net       = MLFFNN(1,128,1)
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3) 
    
    
#########################################################
############## Training the Neural Network ##############
#########################################################
X,Y    = torch.tensor(Xl),torch.tensor(Yl)
def Train_FFNN(No_Episodes):
    MSE        = list()
    for i in range(No_Episodes):
        if (i%50) == 0:
            print("Episode : ", i)
        yp      = net(X)
        loss    = torch.mean((yp-Y)**2)# used  MSE
        MSE.append(loss.item())
        ###########################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), "W_3xe02.pt")  
    return MSE
        
MSE = Train_FFNN(3000)    
np.save("MSE.npy",MSE)
#########################################################
################### Results Plotting ####################
#########################################################

plt.figure(figsize=(9,6))
plt.plot(MSE,color="crimson",label="Mean Square Error")
plt.title("Multilayer Feed Forward Nueral Network - Training - 3*X^2")
plt.xlabel("No of Epochs")
plt.ylabel("MSE")
plt.grid()
plt.legend(loc = "best")
plt.axhline(y=0,color="k")
plt.axvline(x=0, color="k") 
plt.savefig("MSE.jpg",dpi = 420)
plt.show()

#########################################################
#########################################################


    
    
    
    
    
    
    
    
    
    
    