# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:32:16 2020

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
testIn, testOut = InputFile('val.txt')


def sigmoid(m,beta): # op = output, b = beta
    op = 1 / (1 + np.exp(-(beta*m)))
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

def deNormalization(nonNor, nor): #NN output and actual output
    temp = []
    for i in nor:
        t1 = ((max(nonNor) - min(nonNor))*i) + min(nonNor)
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

vip = inputNormalization(testIn)# normalized input #1,2
#print(ip)
vop = normalization(testOut)#normalized output



def committeModel1(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*3 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*3 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,6 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #7,6 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #7,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#6,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,6
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#6,6
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,6
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(7,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,6)))#(7,6)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,6)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(7,1)) + (delta_wo.reshape(7,1)) + alpha*(delta_wo_p.reshape(7,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo

def committeModel2(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*2 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*2 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,4 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #5,4 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #5,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah1 = si.dot(wh1) #[1,3][3,4]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,5
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(5*1) #1,4
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,5
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#4,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,4
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#4,4
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,4
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(5,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,4)))#(7,6)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,4)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(5,1)) + (delta_wo.reshape(5,1)) + alpha*(delta_wo_p.reshape(5,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo



def committeModel3(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*4 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*4 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,8 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #9,8 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #9,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah1 = si.dot(wh1) #[1,3][3 8]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,9
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(9*1) #1,8
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,9
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#8,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,8
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#8,8
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,8
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(9,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,8)))#(9,8)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,8)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(9,1)) + (delta_wo.reshape(9,1)) + alpha*(delta_wo_p.reshape(9,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo



def committeModel4(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*3 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*2 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,6 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #7,4 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #5,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah2 = sh1.dot(wh2) #calculating ah2(7*1) #1,4
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,5
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#6,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,6
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#6,4
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,4
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(7,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,4)))#(7,4)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,6)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(5,1)) + (delta_wo.reshape(5,1)) + alpha*(delta_wo_p.reshape(5,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo


def committeModel5(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*2 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*3 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,4 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #5,6 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #7,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah1 = si.dot(wh1) #[1,3][3 4]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,5
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(5*1) #1,6
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

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#6,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,6
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#4,6
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,6
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(5,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,6)))#(5,6)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,4)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(7,1)) + (delta_wo.reshape(7,1)) + alpha*(delta_wo_p.reshape(7,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo

def committeModel6(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*2 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*4 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,4 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #5,8 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #9,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah1 = si.dot(wh1) #[1,3][3 4]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,5
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(5*1) #1,8
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,9
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#8,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,8
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#4,8
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,4
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(5,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,8)))#(5,8)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,4)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(9,1)) + (delta_wo.reshape(9,1)) + alpha*(delta_wo_p.reshape(9,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo


def committeModel7(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*4 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*2 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,8 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #9,4 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #5,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah1 = si.dot(wh1) #[1,3][3 8]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,9
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(9*1) #1,4
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,5
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#5,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,4
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#8,4
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,8
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(9,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,4)))#(9,4)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,8)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(5,1)) + (delta_wo.reshape(5,1)) + alpha*(delta_wo_p.reshape(5,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo


def committeModel8(ip,op):
    """
    """
    
    nh1 = 1+len(dimensionality[0])*2 # number of hidden layer in 1    
    nh2 = 1+len(dimensionality[0])*2 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,5 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #6,5 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #6,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah1 = si.dot(wh1) #[1,3][3 5]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,6
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(6*1) #1,5
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,6
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#5,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,5
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#5,5
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,5
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(6,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,5)))#(6,5)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,5)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(6,1)) + (delta_wo.reshape(6,1)) + alpha*(delta_wo_p.reshape(6,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo


def committeModel9(ip,op):
    """
    """
    
    nh1 = 1+len(dimensionality[0])*3 # number of hidden layer in 1    
    nh2 = 1+len(dimensionality[0])*3 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,7 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #8,7 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #8,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah1 = si.dot(wh1) #[1,3][3 7]
            
            ##sh1 = output of hidden layer 1
            sh11 = sigmoid(beta,ah1)
            
            #adding sh1_0 (bias)
            sh1=[1]
            sh1.extend(sh11)
            sh1 = np.array(sh1) #1,8
            
          
            ##ah2 = activation value of hidden layer 2
            ah2 = sh1.dot(wh2) #calculating ah2(8*1) #1,7
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,8
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#7,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,7
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#7,7
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,7
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(8,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,7)))#(8,7)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,7)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(8,1)) + (delta_wo.reshape(8,1)) + alpha*(delta_wo_p.reshape(8,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo


def committeModel10(ip,op):
    """
    """
    
    nh1 = len(dimensionality[0])*3 # number of hidden layer in 1    
    nh2 = len(dimensionality[0])*4 # number of hidden layer in 2
    
    eeta = 0.9 # learning rate parameter
    alpha = 0.3# momentum parameter
    beta = 1#sigmoid constant
    
    np.random.seed(0)
    wh1 = np.random.randn(len(dimensionality[0])+1,nh1) #3,6 #random weight value setting for hidden layer1
    wh2 = np.random.randn(nh1+1,nh2) #7,8 #random weight value setting for hidden layer2
    wo =  np.random.randn(nh2+1,1) #9,1

    
    N = len(ip)
    delta_wo_p = np.zeros(np.shape(wo))
    delta_wh2_p = np.zeros(np.shape(wh2))
    delta_wh1_p = np.zeros(np.shape(wh1))
    
 
    for _ in range(100):
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
            ah2 = sh1.dot(wh2) #calculating ah2(7*1) #1,8
            #sh22 = output of hidden layer 2
            sh22 = sigmoid(beta,ah2) 
            #adding sh2_0 (bias)
            sh2=[1]
            sh2.extend(sh22)
            sh2 = np.array(sh2) #1,9
            
            ##ao = activation value of output layer
            ao = sh2.dot(wo) #calculating ao #1,1
            
            ##so = final output
            so = sigmoid(beta,ao)

            
            predY.append(so[0])
            ## error calculation
            e += ((y-so)**2)/2
        e = e / N
        Yp= sum(predY)/len(predY)
        
        
        ### Back Probagation
        ##delo = error at output layer
        delo = (y-Yp)*sigmaGrad(ao,beta)#dimenson is matter in dot product
        # print((delo))
        
        #delh2 = error at hidden layer 2
        wo_nobias = wo[1:,:]#8,1
        delh2 = ((wo_nobias.dot(delo)).T)*(sigmaGrad(ah2,beta))#1,8
        # print(delh2)
        
        #delh1 = error at hidden layer 1
        wh2_nobias = wh2[1:]#6,8
        delh1 = (delh2.dot(wh2_nobias.T))*sigmaGrad(ah1,beta)#1,6
        # print(delh1)
        
        #delta_wo = delta updates of output layer weights
        delta_wo = eeta*(delo*(sh2.T))
        # print(delta_wo)
        
        #delta_wh2 = delta updates of hidden layer 2 weights
        temp = sh1.reshape(7,1)
        delta_wh2 = eeta*((temp).dot(delh2.reshape(1,8)))#(7,8)
        # print(delta_wh2)
        
        #delta_wh1 = delta updates of hidden layer 1 weights
        temp1 = si.reshape(3,1)
        delta_wh1 = eeta*((temp1).dot(delh1.reshape(1,6)))#
        # print(delta_wh1)
        
        ###### UPDATE WEIGHTS####
        
        wo = (wo.reshape(9,1)) + (delta_wo.reshape(9,1)) + alpha*(delta_wo_p.reshape(9,1))
        # print(wo)
        wh2 = wh2 + delta_wh2 + alpha*(delta_wh2_p)
        wh1 = wh1 + delta_wh1 + alpha*(delta_wh1_p)
   
    return wh1,wh2,wo

"""
validation set : for all output model

"""

def validation(wh1,wh2,wo,ip,op,testOut,beta):
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


w11,w21,w31 = committeModel1(ip, op)
w12,w22,w32 = committeModel2(ip, op)
w13,w23,w33 = committeModel3(ip, op)
w14,w24,w34 = committeModel4(ip, op)
w15,w25,w35 = committeModel5(ip, op)
w16,w26,w36 = committeModel6(ip, op)
w17,w27,w37 = committeModel7(ip, op)
w18,w28,w38 = committeModel8(ip, op)
w19,w29,w39 = committeModel9(ip, op)
w110,w210,w310 = committeModel10(ip, op)



predY1,error1,e1 = validation(w11,w21,w31,ip,op,testOut,1)
predY2,error2,e2 = validation(w12,w22,w32,ip,op,testOut,1)
predY3,error3,e3 = validation(w13,w23,w33,ip,op,testOut,1)
predY4,error4,e4 = validation(w14,w24,w34,ip,op,testOut,1)
predY5,error5,e5 = validation(w15,w25,w35,ip,op,testOut,1)
predY6,error6,e6 = validation(w16,w26,w36,ip,op,testOut,1)
predY7,error7,e7 = validation(w17,w27,w37,ip,op,testOut,1)
predY8,error8,e8 = validation(w18,w28,w38,ip,op,testOut,1)
predY9,error9,e9 = validation(w19,w29,w39,ip,op,testOut,1)
predY10,error10,e10 = validation(w110,w210,w310,ip,op,testOut,1)


grandE = (e1+ e2+ e3 + e4 + e5 + e6 +e7 + e8 + e9 +e10)/10

def avgYvalue(predY1, predY2,predY3, predY4, predY5,predY6, predY7, predY8, predY9, predY10):
    grandY = []
    for i in range(len(predY1)):
        temp  = (predY1[i] + predY2[i] + predY3[i] + predY4[i] + predY5[i] + predY6[i]
                 + predY7[i] + predY8[i] + predY9[i] + predY10[i]) / 10
        grandY.append(temp)
    return grandY

def avgEvalue(error1, error2, error3, error4, error5, error6, error7, error8, error9, error10):
    grandE = []
    for i in range(len(error1)):
        temp  = (error1[i]+ error2[i] + error3[i] + error4[i] + error5[i] + error6[i]
        + error7[i] + error8[i] + error9[i] + error10[i])/ 10
        grandE.append(temp)
    return grandE

grandY = avgYvalue(predY1, predY2,predY3, predY4, predY5,predY6, predY7, predY8, predY9, predY10)
grandE = avgEvalue(error1, error2, error3, error4, error5, error6, error7, error8, error9, error10)
# print(grandE)
# print(grandY)

print(deNormalization(testOut, grandY))




plt.figure(figsize=(15,13))
plt.grid()
plt.xlim(-1, 100)
plt.ylim(-0.5, 0.5)
plt.plot(error1)
plt.plot(error2)
plt.plot(error3)
plt.plot(error4)
plt.plot(error5)
plt.plot(error6)
plt.plot(error7)
plt.plot(error8)
plt.plot(error9)
plt.plot(error10)
plt.plot(grandE, linewidth = 5)
plt.xlabel("set of errors for different trials")
plt.ylabel("normalized error range from 0 to 1")
plt.legend(loc= 'best')
plt.show()




plt.figure()
plt.plot(grandY, linewidth = 5)
plt.plot(predY1)
plt.plot(predY2)
plt.plot(predY3)
plt.ylabel("y value")
plt.legend(loc = 'best')
plt.show()














    


















end = time.time()
print("Total time has taken to run this code is ", end - start, "seconds")


        