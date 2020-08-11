# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:01:34 2020

@author: Sri Harsha, Lavanya, Sivaraman
"""
print(__doc__)

import numpy as np, matplotlib.pyplot as plt
import datetime, time

todaDate = datetime.date.today()
start = time.time()

car = np.load('car.npy')
horse = np.load('horse.npy')
plane = np.load('plane.npy')
ship = np.load('ship.npy')
truck = np.load('truck.npy')

#print(len(car), len(horse),len(plane),len(ship),len(truck))
#
print(len(car[0]))
print(len(car))
#print("{}".format(ship))

from numpy import asarray
from numpy import savetxt

#carr= car[0][0]
#
#savetxt('car.csv',carr, delimiter=',')
##c = np.savetxt('car.txt', car) 
##































end = time.time()
print("Total time to run this code", end-start, "Seconds")



