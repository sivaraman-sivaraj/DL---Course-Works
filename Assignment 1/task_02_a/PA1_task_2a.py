# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:01:34 2020

@author: Sri Harsha, Lavanya, Sivaraman.S
"""
print(__doc__)

import numpy as np, matplotlib.pyplot as plt
import datetime, time

todaDate = datetime.date.today()
print("The code file is run on ", todaDate,'\n \n')
start = time.time()


def InputFile(filename):
    datafile = open(filename,'r')
    inputVector = []
    label = []
    datafile.readline()
    for line in datafile:
        line.strip()
        a,b,c = line.split(",")
        inputVector.append([float(a),float(b)])
        c.split('\n')
        label.append(c[0])
    datafile.close()
    return inputVector,label

dimensionality,label = InputFile('traingroup6.csv')


def sigmoid(b,m): # op = output, b = beta
    op = 1 / (1 + np.exp(-(b*m)))
    return op

def sigmaGrad(m,beta):
    op = np.exp(-(beta*m)) / ((1 + np.exp(-(beta*m)))**2)
    return op*beta

def normalization(elements):
    temp = []
    for i in elements:
        t1 = (i - min(elements))/(max(elements) - min(elements))
        temp.append(t1)
    return temp

def softmax(a): ##where a is row or column vector
    temp = []
    for i in a:
        temp.append(np.exp(i))
    dem = sum(temp)
    temp1 = []
    for j in temp:
        temp1.append(j/dem)
    return temp1

    

def inputNormalization(dimensionality):
    L = []
    M = []
    N = []
    for j in range(len(dimensionality[0])):
        temp = []
        L.append(temp)
        
    for i in dimensionality:
        for k in range(len(dimensionality[0])):
            L[k].append(i[k])
            
    for l in L:
        temp = normalization(l)
        M.append(temp)
    
    for j in range(len(M[0])):
        temp = []
        for i in range(len(M)):
            temp.append(M[i][j])
        N.append(temp)
    return N

ip = inputNormalization(dimensionality)# normalized input #1,2
#print(ip)
N = len(dimensionality)
opIntialize = np.zeros((N,3))


def outputNormalization(N,op, label):
    for i in range(N):
        if label[i] == '1':
            op[i][1] = 1
        elif label[i] == '2':
            op[i][2] = 2
    return op

op = outputNormalization(N,opIntialize, label)
#print(op)         
    
    

nh1 = len(dimensionality[0])*3 # number of hidden layer in 1    
nh2 = len(dimensionality[0])*3 # number of hidden layer in 2

eeta = 0.9 # learning rate parameter
alpha = 0.3# momentum parameter
beta = 1#sigmoid constant

np.random.seed(0)
wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,6 #random weight value setting for hidden layer1
#print(wh1)
wh2 = np.random.randn(nh1+1,nh2) #7,6 #random weight value setting for hidden layer2
wo =  np.random.randn(nh2+1,3) #7,3 output as 3

def gradientupdate(ip,op,wh1,wh2,wo,eeta,alpha,beta):
    """
    % the function updates weights using generalized update rule
    % ip - input
    % op - output
    % wh1, wh2, wo - weights of hidden layer 1, 2, and output layer
    % eeta - learning rate
    % alpha - momentum
    %N = Number of examples

    """
    N = len(ip)
    ##Initialize delta_weights of previous example for generalized update rule
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    error = []
    
    for j in range(2):
        e = 0 # temporary error declaration
        predL =[] #predicted label
        for i in range(N):
            #x = n-th example
            xtemp = ip[i]
            
            x = []
            for m in xtemp:
                x.append(round(m,8))
            print(x)
            # t = target output of nth example
            y = op[i]
             #si = output of input layer  
            sit = [1]
            si = sit.append(x[0])
            si = sit.append(x[1])
            sif = np.array(si)
            print((sif))
            #ah1 = activation value of hidden layer 1
            
            ah1 = sif.dot(wh1) #[1,3][3 6]
            print(ah1)
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1 = np.hstack((sh1,sh11))
            
            sh1 = np.array(sh1)
#            print(sh1)
            sh1f = sh1.reshape(1,7)
            
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2[1,7][7,6] = [1,6]
            
            
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2)
            
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,7
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #[1,7][7,3]=[1,3]
            
            ##so = final output
            so = softmax(ao) ## softmax neuron
            
            arN = np.argmax(y) # argmx in output value
            predL.append(arN)
            
            e = e - np.log(so[arN])
            
            e = e/N
            """
            back probagation part
            """
            ### Back Probagation
            ##delo = error at output layer
            delo =np.array(so)#dimenson is matter in dot product
            delo[arN] = delo[arN] + 1 #[1,3]
            
            
            #delh2 = error at hidden layer 2
            wo_nobias = wo[1:,:]#6,3
            delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))# [6,3][3,1] = [6,1] as [1,6]
            
        
            #delh1 = error at hidden layer 1
            wh2_nobias = wh2[1:]#6,6
            delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,6
            

            #delta_wo = delta updates of output layer weights
            xd1 = delo.reshape(3,1)
            xd2 = sh2.reshape(1,7)
            delta_wo = eeta*(xd1.dot(xd2))
            
            
            #delta_wh2 = delta updates of hidden layer 2 weights
            temp = sh1.reshape(7,1)
            delta_wh2 = eeta*((temp).dot(delh2.reshape(1,6)))#(7,6)
           
           
            #delta_wh1 = delta updates of hidden layer 1 weights
            temp1 = si.reshape(3,1)
            delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,6)))#
            
            ###### UPDATE WEIGHTS####
            wo = (wo.reshape(7,3)) + (delta_wo.reshape(7,3)) + alpha*(delta_wo_p.reshape(7,3))
#            print(wo)
            
            wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
            wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
            
            ##storing delta weights for next example
            delta_wo_p = delta_wo
            delta_wh2_p = delta_wh2
            delta_wh1_p = delta_wh1
        error.append(e)
    return (wh1,wh2,wo,error,predL)

            

a,b,c,d,e = gradientupdate(ip,op,wh1,wh2,wo,eeta,alpha,beta)























    




































end = time.time()
print("Total time to run this code",end-start, "Seconds")



