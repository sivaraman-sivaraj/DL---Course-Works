# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:01:34 2020

@author: Sri Harsha, Lavanya, Sivaraman
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
    output = []
    datafile.readline()
    for line in datafile:
        a,b,c = line.split(" ")
        inputVector.append([float(a),float(b)])
        output.append(float(c))
    datafile.close()
    return inputVector,output

dimensionality, output = InputFile('train100.txt')

#print(output)

eeta = 0.6 # learning rate parameter
alpha = 0.01# momentum parameter
beta = 0.4

    
def sigmoid(b,m): # op = output, b = beta
    op = 1 / (1 + np.exp(-(b*m)))
    return op


def matSig(lis, beta):#sigmoid function implement
    F = []
    for i in lis:
        temp = sigmoid(beta,i)
        F.append(temp)
    return F

def sigmgrad(m):
    t1 = []
    for i in m:
        t2=[]
        for j in i:
            temp = np.exp(-j) / ((1 + np.exp(-j))**2)
            t2.append(temp)
        t1.append(t2)
    return t1


def normalization(elements):
    temp = []
    for i in elements:
        t1 = (i - min(elements))/(max(elements) - min(elements))
        temp.append(t1)
    return temp

L1 =[]
L2 =[]
  
for i in dimensionality:
    L1.append(i[0])
    L2.append(i[1])
    
nL1 = normalization(L1)#first element of normalized input vector
nL2 = normalization(L2)#second element of normalized input vector
nOutput = normalization(output)#normalized output


def fDv(nL1,nL2): #final normalized dimensional vector
    temp = []
    for i in range(len(nL1)):
        t1 = [nL1[i],nL2[i]]
        temp.append(t1)
    return temp

ip = fDv(nL1,nL2)
nh1 = len(dimensionality[0])*3         
nh2 = len(dimensionality[0])*3

np.random.seed(0)
wh1 = np.random.randn(nh1, len(dimensionality[0])+1)#random weight value setting for hidden layer1
wh2 = np.random.randn(nh2, nh1+1)#random weight value setting for hidden layer2
wo =  np.random.randn(1,nh2+1)


print(wh1)
avgError = 0
errorDiff = 0


def gradientUpdate(ip,nOutput,wh1,wh2,wo,eeta,alpha,beta):
    
    N = len(ip)
    delta_wo_p = np.zeros(len(wo))
    delta_wh2_p = np.zeros(len(wh2))
    delta_wh1_p = np.zeros(len(wh1))
#    iteration = np.arange(0,5000)
    error = []
    for i in range(5000):
        e = 0 # temporary error declaration
        predY =[]
        for j in range(N):
            
            y = nOutput[j]
            x = ip[j] 
            aat=[1]
            aat.extend(x)
            s1 = np.transpose(aat)
            
            #ah1 = activation value of hidden layer 1
            
            ah1 = np.matmul(wh1,s1)
            print(ah1)
            #sh11 = output of hidden layer 1
            sh11 = matSig(beta,ah1)#calculating sh1
            aat1 = [1]
            aat1.extend(sh11)
            sh1 = np.transpose(sh11)
            
            #ah2 = activation value of hidden layer 2
            wh2 = np.array(wh2)
            ah2 = wh2.dot(sh11.transpose())
            
            #sh2 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) #% calculating sh2
            aat2 = [1]
            aat2.extend(sh22)
            sh2 = np.transpose(aat2)
            
            #ao = activation value of output layer
            ao = wo.dot(sh2)#% calculating ao
            
            #so = final output
            ftt1 = np.array(beta*ao)
            so = sigmoid(ftt1) # calculating so
            
            predY.append(so)
            
            #average error calculation
            e = e + (1/(2*N))*((y - so)**2)
            
            #### BACK PROPOGATION #####
            # delo = error at output layer
            delo = (y - so)*sigmgrad(beta*ao)
            delo = beta*delo
            
            #delh2 = error at hidden layer 2
            wo_nobias = wo[1:]
            delh2 = beta*((wo_nobias*delo)*sigmgrad(beta*ah2))
            
            #delh1 = error at hidden layer 1
            wh2_nobias = wh2[1:]
            delh1 = beta*((wh2_nobias*delh2)*sigmgrad(beta*ah1))
            
            
            #delta_wo = delta updates of output layer weights
            delta_wo = eeta*(delo*sh2)
            
            #delta_wh2 = delta updates of hidden layer 2 weights
            delta_wh2 = eeta*(delh2*sh1)
            
            #delta_wh1 = delta updates of hidden layer 1 weights
            delta_wh1 = eeta*(delh1*s1)
            
            
            ###### UPDATE WEIGHTS####
            wo = wo + delta_wo + alpha*(delta_wo_p)
            wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
            wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
            
            ##storing delta weights for next example
            delta_wo_p = delta_wo
            delta_wh2_p = delta_wh2
            delta_wh1_p = delta_wh1
        error.append(e)
    return (wh1,wh2,wo,error,predY)

wh1,wh2,wo,error,predY = gradientUpdate(ip,nOutput,wh1,wh2,wo,eeta,alpha,beta)




            
        
         
        
        
        





            
    
        
        
        
































end = time.time()
print("\n \n Total time to run this code", end-start, "Seconds")



