# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:32:26 2020

@author: sivaraman sivaraj
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
testIn, testOut = InputFile('val.txt')

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
op = normalization(output)#normalized output
nh1 = len(dimensionality[0])*3 # number of hidden layer in 1    
nh2 = len(dimensionality[0])*3 # number of hidden layer in 2

eeta = 0.9 # learning rate parameter
alpha = 0.3# momentum parameter
beta = 1#sigmoid constant

np.random.seed(0)
wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,6 #random weight value setting for hidden layer1
wh2 = np.random.randn(nh1+1,nh2) #7,6 #random weight value setting for hidden layer2
wo =  np.random.randn(nh2+1,1) #7,1
print(wh1)

def gradientUpdate(ip,op,wh1,wh2,wo,eeta,alpha,beta):
    """
    """
    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
    error = []
    for j in range(1):
        e = 0 # temporary error declaration
        predY =[]
        for i in range(N):
            
            #x = n-th example
            x = ip[i]
            # t = target output of nth example
            y = op[i]
            #si = output of input layer  
            si = [1]
            si.extend(x)
            si = np.array(si)
            
            #ah1 = activation value of hidden layer 1
            ah1 = si.dot(wh1) #[1,3][3 6]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,7
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(7*1) #1,6
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,7
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so)
            ## error calculation
            e += ((y-so)**2)/2
#            print(e)
            e = e/N
            
            op = np.array(op)
            erMean = (sum(op-predY))/len(op)
        
            ### Back Probagation
            ##delo = error at output layer
            delo = (y-so)*sigmaGrad(ao,beta)#dimenson is matter in dot product
#           print((delo))
        
#           #delh2 = error at hidden layer 2
            wo_nobias = wo[1:,:]#6,1
            delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,6
            
#        
#            #delh1 = error at hidden layer 1
            wh2_nobias = wh2[1:]#6,6
            delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,6
#            print(delh1)
#        
        #delta_wo = delta updates of output layer weights
            delta_wo = eeta*(delo*(sh2.T))
#            print(delta_wo)
            
#        #delta_wh2 = delta updates of hidden layer 2 weights
            temp = sh1.reshape(7,1)
            delta_wh2 = eeta*((temp).dot(delh2.reshape(1,6)))#(7,6)
           # print(delta_wh2)
#            
#        #delta_wh1 = delta updates of hidden layer 1 weights
            temp1 = si.reshape(3,1)
            delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,6)))#
 #           print(delta_wh1)
#        
        ###### UPDATE WEIGHTS####
            wo = (wo.reshape(7,1)) + (delta_wo.reshape(7,1)) + alpha*(delta_wo_p.reshape(7,1))
#            print(wo)
            
            wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
            wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
#            
        ##storing delta weights for next example
            delta_wo_p = delta_wo
            delta_wh2_p = delta_wh2
            delta_wh1_p = delta_wh1
        error.append(e)
    return (wh1,wh2,wo,error,predY)
            
   
wh1t, wh2t,wot,errort,predYt = gradientUpdate(ip,op,wh1,wh2,wo,eeta,alpha,beta)       
        
 
            
            
def validation(wh1,wh2,w0,ip,op):
    N = len(ip)
    
    e = 0 # temporary error declaration
    error = []
    predY =[]
    for i in range(N):
        
        #x = n-th example
        x = ip[i]
        # t = target output of nth example
        y = op[i]
        #si = output of input layer  
        si = [1]
        si.extend(x)
        si = np.array(si)
        
        #ah1 = activation value of hidden layer 1
        ah1 = si.dot(wh1) #[1,x][x n]
        
        ##sh1 = output of hidden layer 1
        sh11 = sigmoid(beta,ah1)
        
        #adding sh1_0 (bias)
        sh1=[1]
        sh1.extend(sh11)
        sh1 = np.array(sh1) #1,x
        
      
        ##ah2 = activation value of hidden layer 2
        ah2 = sh1.dot(wh2) #calculating ah2(x*1) #1,n
        #sh22 = output of hidden layer 2
        sh22 = sigmoid(beta,ah2) 
        #adding sh2_0 (bias)
        sh2=[1]
        sh2.extend(sh22)
        sh2 = np.array(sh2) #1,n
        
        ##ao = activation value of output layer
        ao = sh2.dot(wo) #calculating ao #1,1
        
        ##so = final output
        so = sigmoid(beta,ao)

        
        predY.append(so[0])
        ## error calculation
        e += ((y-so)**2)/2
        error.append(y-so[0])

    e = e/N
    return predY,error,e


a,b,c = validation(wh1t, wh2t,wot,testIn,testOut)

print(a,b,c)

    
    
    





















end = time.time()
print("Total time has taken to run this code is ", end - start, "seconds")



