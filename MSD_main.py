import os
import numpy as np 
import matplotlib.pyplot as plt 

def Dynamics_MSD(F,X):
    """
    X : it should be (2*1) matrix in numpy array format
    F : scalar value (real number)
    """
    dt = 0.001 # time step
    #######################
    m = 1
    c = 0.5
    k = 1
    #######################
    a11 = 0
    a12 = 1
    a21 = -k/m 
    a22 = -c/m 
    
    b1 = 0
    b2 = -1/m
    
    A = np.array([[a11,a12],[a21,a22]]) 
    B = np.array([[0],[-1/m]]) 
    ########################
    X_dot = A.dot(X) + (B*F) 
    
    X_nxt = X + (X_dot*dt) 
    return X_nxt.tolist() 

#########################################################
################### Sine Wave Input #####################
#########################################################
T = np.arange(0,30000)*0.001
F = np.sin(T)*0.5
#########################################
############## Simulation ###############
#########################################
# Xs,Xds = [2],[0] # set your initial condition here
# X_ip = np.array([[Xs[-1]],[Xds[-1]]])
# for i in range(len(F)):
#     F_ip  = F[i] 
#     X_nxt = Dynamics_MSD(F_ip,X_ip) 
#     ### updating the state 
#     Xs.append(X_nxt[0][0]) 
#     Xds.append(X_nxt[1][0]) 
#     X_ip = np.array([[Xs[-1]],[Xds[-1]]]) 
#     ###

# plt.figure(figsize=(10,4))
# plt.plot(T,Xs[0:len(Xs)-1],color="crimson",label="X value")
# plt.grid() 
# plt.xlabel("Time (seconds)") 
# plt.ylabel("Amplitude") 
# plt.title("Mass Spring Damper System - Simulation")
# plt.show() 
##########################################################
################## PID Controller ########################
##########################################################
def PID_control(X_deisre,Xd_desire,X_actual,Xd_actual):
    Kp = 10
    Kd = 2
    E_x  = X_actual  - X_deisre 
    E_xd = Xd_actual - Xd_desire 
    U  = (Kp*E_x) + (Kd*E_xd) 
    return U 
###########################################################
X_desire  = np.sin((np.pi/5)*T) 
Xd_desire = np.gradient(X_desire) 

##### PID COntroller Simulation
Xs,Xds = [2],[0] # set your initial condition here
X_ip = np.array([[Xs[-1]],[Xds[-1]]]) 

for i in range(len(X_desire)):
    X_des_temp  = X_desire[i] 
    Xd_des_temp = Xd_desire[i] 
    F_ip  = PID_control(X_des_temp,Xd_des_temp,Xs[-1],Xds[-1])
    X_nxt = Dynamics_MSD(F_ip,X_ip) 
    ### updating the state 
    Xs.append(X_nxt[0][0]) 
    Xds.append(X_nxt[1][0]) 
    X_ip = np.array([[Xs[-1]],[Xds[-1]]])  
    
    
plt.figure(figsize=(10,4)) 
plt.plot(T,Xs[0:len(Xs)-1],color="teal",label="X Actual") 
plt.plot(T,X_desire,color="crimson",label="X value",linestyle="--")
plt.grid() 
plt.legend(loc="best")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (meter)")
plt.show()
###########################################################
###########################################################
