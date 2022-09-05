import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
####################################################################
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
net.load_state_dict(torch.load("W_3xe02.pt"))
net.eval() 


##################################################################
################# Validation Data ################################
##################################################################
import numpy as np
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

Xl,Y_actual = f()


def Validation(X):
    Y_P = list()
    for i in range(len(X)):
        temp = torch.tensor([X[i]])
        op   = net(temp)
        Y_P.append(op.item())
    return Y_P 

Y_Predicted = Validation(Xl)

##################################################################
#################### Results Plotting ############################
##################################################################

plt.figure(figsize=(9,6))
plt.plot(Xl,Y_actual,color = "r",linestyle="--",linewidth = 6.0,label="Actual Points")
plt.plot(Xl,Y_Predicted,color = "g",linewidth = 4.0,label="predicted Points")
plt.title("Results Comparision for the model ($3X^2$)")
plt.xlabel("X") 
plt.ylabel("Y")
plt.axhline(y=0, color="k")
plt.axvline(x=0, color = "k") 
plt.grid()
plt.legend(loc = "best") 
plt.savefig("Results_Comparision.jpg",dpi = 720)
plt.show()
##################################################################
##################################################################


