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

Xl,Yl = f(500)

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
net.load_state_dict(torch.load("W.pt"))
net.eval() 
#######################################################
################### Confusion Matrix ##################
#######################################################

def Confusion_Matrix(X,Y):
    CM = np.zeros((4,4))
    for i in range(len(X)):
        x_temp  = torch.tensor(X[i])
        y_pt    = net(x_temp).tolist()
        y_p     =  np.argmax(y_pt)
        y_a     = np.argmax(Y[i])
        if y_p == y_a:
            CM[y_p][y_p] += 1
        else:
            CM[y_p][y_a] += 1
    return CM

ss = Confusion_Matrix(Xl,Yl)
print(ss)
#######################################################
#######################################################




