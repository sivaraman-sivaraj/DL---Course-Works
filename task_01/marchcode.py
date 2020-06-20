# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 08:18:42 2020

@author: Sivaraman Sivaraj
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

def sigmoid(b,m): # op = output, b = beta
    op = 1 / (1 + np.exp(-(b*m)))
    return op

def sigmaGrad(m,beta):
    op = np.exp(-(beta*m)) / ((1 + np.exp(-(beta*m)))**2)
    return op*beta

def matSig(lis, beta):#sigmoid function implement
    F = []
    for i in range(6):
        temp = sigmoid(beta,lis[i])
        F.append(temp)
    return F

def matSigmgrad(m,beta):
    t1 = []
    for i in range(len(m)):
        temp = sigmaGrad(m[i],beta)
        t1.append(temp)
    return t1

def normalization(elements):
    temp = []
    for i in elements:
        t1 = (i - min(elements))/(max(elements) - min(elements))
        temp.append(t1)
    return temp

def inputNormalization(dimensionality):
    L1 =[]
    L2 =[]
    for i in dimensionality:
        L1.append(i[0])
        L2.append(i[1])
    nL1 = normalization(L1)#first element of normalized input vector
    nL2 = normalization(L2)#second element of normalized input vector
    temp = []
    for i in range(len(nL1)):
        t1 = [nL1[i],nL2[i]]
        temp.append(t1)
    return temp


ip = inputNormalization(dimensionality)
nOutput = normalization(output)#normalized output

nh1 = len(dimensionality[0])*3 # number of hidden layer in 1    
nh2 = len(dimensionality[0])*3 # number of hidden layer in 2
np.random.seed(0)

def randomWeight(m,n):
    temp = []
    for i in range(m):
        temp1 = []
        for j in range(n):
            nor = np.random.randn(1)
            temp1.append(nor[0])
        temp.append(temp1)
    return temp

def matmulp(q1,q2):##consider q1, q2 as 1*6 matrix
    ans = []
    for i in range(6):
        row =[]
        for j in range(6):
            temp = q1[i]*q2[j]
            row.append(temp)
        ans.append(row)
    return ans
    

wh1 = randomWeight(nh1, len(dimensionality[0])+1)#random weight value setting for hidden layer1
wh2 = randomWeight(nh2, nh1+1)#random weight value setting for hidden layer2
wo =  randomWeight(1,nh2+1)#random weight value setting for output layer
#rint(wh2[1:])
avgError = 0
errorDiff = 0
eeta = 0.6 # learning rate parameter
alpha = 0.01# momentum parameter
beta = 0.4

def toUpdatewh2(wh2):
    ans =[]
    for i in wh2:
        g = i[1:]
        ans.append(g)
    return ans

print(toUpdatewh2(wh2))

def gradientUpdate(ip,nOutput,wh1,wh2,wo,eeta,alpha,beta):
    N = len(ip)
    delta_wo_p = np.zeros(len(wo))
    delta_wh2_p = np.zeros(len(wh2))
    delta_wh1_p = np.zeros(len(wh1))
    error = []
    
    for j in range(5000):
        e = 0 # temporary error declaration
        predY =[]
        for i in range(N):
            #x = n-th example
            x = ip[i]
            # t = target output of nth example
            y = nOutput[i]
            #si = output of input layer  
            si = [1]
            si.extend(x)
            #ah1 = activation value of hidden layer 1
            ah1 = np.matmul(wh1,si)
#            print(ah1[0])
            ##sh1 = output of hidden layer 1
            sh11 = matSig(ah1,beta)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
                        
            ##ah2 = activation value of hidden layer 2
            ah2 = np.matmul(wh2,sh1) #calculating ah2
            
            #sh22 = output of hidden layer 2
            sh22 = matSig(ah2,beta) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            
            ##ao = activation value of output layer
            ao = np.matmul(wo,sh2) #calculating ao
            
            ##so = final output
            so = sigmoid(ao,beta)
            
            predY.append(so)
            
            ## error calculation
            e += ((y-so)**2)/2
        e = e/N
#        print(e)
        ### Back Probagation
        ##delo = error at output layer
        delo = (y- so)*sigmaGrad(ao, beta)
        delo = beta*delo
        
        ##delh2 = error at hidden layer 2
        wo_nobias = wo[0][1:]
        q1 = np.array(beta*((wo_nobias*delo)))
        q2 =np.array( matSigmgrad(ah2, beta))
        delh2 = matmulp(q1,q2)
        
        ##delh1 = error at hidden layer 1
        wh2_nobias = toUpdatewh2(wh2)
        

        
        
            
            
    

#zz = gradientUpdate(ip,nOutput,wh1,wh2,wo,eeta,alpha,beta)



















