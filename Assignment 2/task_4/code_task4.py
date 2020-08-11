"""
Created on Tue May 11 12:03:25 2020

@author: Sivaraman Sivaraj
"""
import numpy as np
import matplotlib.pyplot as plt

# sigmoid function
def sigm(a):
    s = 1 / (np.exp(-a) + 1)
    return s

# Defining RBM class
class RBM(object):
    def __init__(self,nv,nh): #nv - # visible layer nodes, nh - # hidden layer nodes
        self.w = np.random.randn(nh,nv) # w - weight matrix
        self.b = np.random.randn(1,nv) # b - visible layer bias
        self.c = np.random.randn(1,nh) # c - hidden layer bias
        self.nh = nh

    def solve_for_hl(self,v): # calculate hidden layer state given visible layer state (v)
        ah = np.dot(v,self.w.T) + self.c # activation value
        ah = sigm(ah) # calculate probablities
        h = np.random.binomial(1, ah) # calculate state from probabilities
        h = h.astype('float32')
        return h,ah


    def solve_for_vl(self,h): # calculate visible layer state given hidden layer state (v)
        av = np.dot(h,self.w) + self.b #activation value
        av = sigm(av) # calculate probablities
        v = np.random.binomial(1, av) # calculate state from probabilities
        v = v.astype('float32')
        return v,av

    def update_params(self,eeta,vo,vk,aho,ahk): # gradient ascent method to update parameters
        self.b += eeta*np.sum(vo - vk,axis=0)
        self.c += eeta*np.sum(aho - ahk,axis=0)
        self.w += eeta*(np.dot(aho.T,vo) - np.dot(ahk.T,vk))

    def get_params(self): # return parameters
        return self.w,self.b,self.c

    def set_params(self,w,b,c): # set parameters
        self.w = w
        self.b = b
        self.c = c

    def fitmodel(self,x,n_epoch,batch_size,eeta): # training given input x
        # n_epoch - number of epoches
        # batch_size = minibatch size
        # eeta = learning rate
        N = x.shape[0] # N = number of samples
        for epoch in range(n_epoch):
            for n in range(0,N-batch_size+1,batch_size):
                vo = x[n:n+batch_size,:]
                _,aho = self.solve_for_hl(vo)#get probabilities of hiddenlayer state
                
                #k step contrastive divergence, k = 2
                vk = x[n:n+batch_size,:]
                for k in range(2):
                    hk,_ = self.solve_for_hl(vk)
                    vk,_ = self.solve_for_vl(hk)
                _,ahk = self.solve_for_hl(vk) #get probabilities of hidden layer state given vk

                self.update_params(eeta,vo,vk,aho,ahk) #update parameters using gradient ascent


    def testmodel(self,x): # calculate error of the RBM model for the given input x
        N = x.shape[0] # Number of samples
        e = 0
        for n in range(N):
            vo = x[n,:]
            h1,_ = self.solve_for_hl(vo) # calculate hidden layer state
            v1,_ = self.solve_for_vl(h1) # calculate visible layer state
            e += (np.mean(np.abs(vo - v1))) # calculate error
        return e/N

    def get_hiddenlayer(self,x): # get hidden layer states for given input x
        N = x.shape[0]
        h = np.empty([N,self.nh])
        for n in range(N):
            vo = x[n,:]
            h[n,:],_ = self.solve_for_hl(vo) # calculate hidden layer state
        return h
    

# getting training data
x1 = np.load('x1.npy')
x = x1[:27000,:] / 255
v = (x >= 0.5)
v = v.astype('float32')
# building RBM stack 1
nv = v.shape[1] # number of visible layer nodes
nh = 262 # number of hidden layer nodes
stack1 = RBM(nv,nh)
batch_size = 10
eeta = 0.005
err_train = []
err_train.append(stack1.testmodel(v))
for i in range (10):
    stack1.fitmodel(v,1,batch_size,eeta)
    e = stack1.testmodel(v)
    if ((e <= 0.215) and (abs(e-err_train[-1]) <= 0.001)):
        break
    err_train.append(e)
err_train.append(e)
plt.plot(err_train)
h1 = stack1.get_hiddenlayer(v)
w1,b1,c1 = stack1.get_params()



# building RBM stack 2
nv2 = h1.shape[1] #number of visible layer nodes for stack 2
nh2 = 130 #number of hidden layer nodes for stack 2
stack2 = RBM(nv2,nh2)
batch_size2 = 10
eeta2 = 0.005
err_train2 = []
err_train2.append(stack2.testmodel(h1))
for i in range (20):
    stack2.fitmodel(h1,1,batch_size2,eeta2)
    e = stack2.testmodel(h1)
    if ((e <= 0.014) and (abs(e-err_train2[-1]) <= 0.0002)):
        break
    err_train2.append(e)
err_train2.append(e)
plt.plot(err_train2)
h2 = stack2.get_hiddenlayer(h1)
w2,b2,c2 = stack2.get_params()


#building RBM stack 3
nv3 = h2.shape[1] #number of visible layer nodes for stack 3
nh3 = 65 #number of hidden layer nodes for stack 3
stack3 = RBM(nv3,nh3)
batch_size3 = 10
eeta3 = 0.005
err_train3 = []
err_train3.append(stack3.testmodel(h2))
for i in range (10):
    stack3.fitmodel(h2,1,batch_size3,eeta3)
    e = stack3.testmodel(h2)
    if ((e <= 0.112) and (abs(e-err_train3[-1]) <= 0.001)):
        break
    err_train3.append(e)
err_train3.append(e)
plt.plot(err_train3)
w3,b3,c3 = stack3.get_params()





"Classifiers training"


import torch
import torch.nn as nn
import torch.optim as optim


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




"Getting data"

x1 = np.load('x1.npy') #input dataset
v = x1 / 255
v = v.astype('float32')
y1 = np.load('y1.npy') #target classes corresponding to input dataset
"Converting the input and target data corresponding to training and validation torchtensors"


ip_train = torch.from_numpy(v[:300,:]) #part A - fine tuning
ip_pt = torch.from_numpy(v[300:27000,:]) #part B - pre training
ip_val = torch.from_numpy(v[27000:,:]) #Part C - validation
op_train = torch.from_numpy(y1[:300])
op_pt = torch.from_numpy(y1[300:27000])
op_val = torch.from_numpy(y1[27000:])
op_train = op_train.type(torch.LongTensor)
op_pt = op_pt.type(torch.LongTensor)
op_val = op_val.type(torch.LongTensor)


"converting the weights from pre-training to torch tensors"
fc1_w,fc1_b = torch.from_numpy(w1),torch.from_numpy(c1.flatten())
fc2_w,fc2_b = torch.from_numpy(w2),torch.from_numpy(c2.flatten())
fc3_w,fc3_b = torch.from_numpy(w3),torch.from_numpy(c3.flatten())
fc1_w,fc1_b = fc1_w.type(torch.FloatTensor),fc1_b.type(torch.FloatTensor)
fc2_w,fc2_b = fc2_w.type(torch.FloatTensor),fc2_b.type(torch.FloatTensor)
fc3_w,fc3_b = fc3_w.type(torch.FloatTensor),fc3_b.type(torch.FloatTensor)
# building neural network for classification task



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(nv,nh)
        self.fc2 = nn.Linear(nh,nh2)
        self.fc3 = nn.Linear(nh2,nh3)
        self.fc4 = nn.Linear(nh3,5)

    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.softmax(self.fc4(x),dim=1)
        return x

net = Net() #creating the instance of the neural network


"Using the pre-trained weights to initialize the weights of neural network"
with torch.no_grad():
    net.fc1.weight.data = fc1_w
    net.fc1.bias.data = fc1_b
    net.fc2.weight.data = fc2_w
    net.fc2.bias.data = fc2_b
    net.fc3.weight.data = fc3_w
    net.fc3.bias.data = fc3_b

    
criterion = nn.CrossEntropyLoss() # cross entropy loss function
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9) #stochastic gradient descent
N_train = ip_train.shape[0] #number of training examples
N_val = ip_val.shape[0] #number of validation examples
minibatch = 10 #minibatch size
err = []
for epoch in range(1000):
    e = 0
    for n in range(0,N_train-minibatch+1,minibatch):
        ip = ip_train[n:n+minibatch,:]
        t = op_train[n:n+minibatch]
        optimizer.zero_grad()
        op = net(ip)
        loss = criterion(op,t)
        loss.backward()
        optimizer.step()
        e += loss.item()
    if epoch >= 1:
        if ((e*minibatch/N_train <= 0.95) and (abs(e*minibatch/N_train - err[-1])<= 0.001)):
            break
    err.append(e*minibatch/N_train)
err.append(e*minibatch/N_train)
plt.plot(err)
with torch.no_grad():
    outputs = net(ip_train)
    out_pt = net(ip_pt)
    out_val = net(ip_val)

_, predicted = torch.max(outputs,1) #predicting the classes for fine-tuning data
_,pred_pt = torch.max(out_pt,1) #predicting the classes for pre-training data
_,pred_val = torch.max(out_val,1) #predicting the classes for validation data
conf_train = calc_confmat(op_train,predicted,5) #confusion matrix for finetuning data
conf_pt = calc_confmat(op_pt,pred_pt,5) #confusion matrix for pre-training data
conf_val = calc_confmat(op_val,pred_val,5) #confusion matrix for validation data
acc_train = torch.sum(torch.diag(conf_train)) / torch.sum(conf_train) #accuracy for fine-tuning
acc_pt = torch.sum(torch.diag(conf_pt)) / torch.sum(conf_pt) #accuracy for pretraining
acc_val = torch.sum(torch.diag(conf_val)) / torch.sum(conf_val) #accuracy for validation


