# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 01:50:14 2020
!!!!!!IMPORTANT!!!!!!!!
Each part seperated from each other. Please run cell by cell
@author: Abdullah Hamza Şahin
"""
#%%
import numpy as np
from matplotlib import pyplot as plt

#####################################
############ DIRECT METHOD ##########
#####################################
print("Direct Method")

def getPoweredX(deg, X_t):
    X_pow = X_t.copy()
    for i in range(2, deg+1):
        powmat = (X_t[:,1]**i).reshape(X_pow.shape[0], 1)
        X_pow = np.append(X_pow, powmat, axis=1)
    return X_pow

def calculateEmpricalRisk(yh):
    error = sum((Y - yh)**2)
    return error/len(Y)
    

X = np.array([31,33,31,49,53,69,101,99,143,132,109])
X = X.reshape(X.shape[0],1)
X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)

Y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
Y = Y.reshape(Y.shape[0],1)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

#For degree 1 polynomial
deg1 = 1
X1 = getPoweredX(deg1, X_train)

first_part = np.linalg.inv(np.dot(X1.transpose(),X1))
second_part = np.dot(first_part, X1.transpose())
W1 = np.dot(second_part, Y)

points = np.linspace(20,160,200).reshape(200,1)
points = np.append(np.ones((200, 1)), points, axis=1)
points = getPoweredX(deg1,points)

ax1.plot(X,Y, "ro")
ax1.axis([0, 160, 0, 1800])
ax1.get_xaxis().set_visible(False)
ax1.plot(points, np.dot(points, W1))
ax1.set_title("Degree=1")
print("Empirical risk for degree 1 model: ", calculateEmpricalRisk(np.dot(X1,W1)))

#For degree 2 polynomial
deg2 = 2
X2 = getPoweredX(deg2, X_train)

first_part = np.linalg.inv(np.dot(X2.transpose(),X2))
second_part = np.dot(first_part, X2.transpose())
W2 = np.dot(second_part, Y)

points = np.linspace(20,160,200).reshape(200,1)
points = np.append(np.ones((200, 1)), points, axis=1)
points = getPoweredX(deg2,points)

ax2.plot(X,Y, "ro")
ax2.axis([0, 160, 0, 1800])
ax2.get_xaxis().set_visible(False)
ax2.plot(points, np.dot(points, W2))
ax2.set_title("Degree=2")
print("Empirical risk for degree 2 model: ", calculateEmpricalRisk(np.dot(X2,W2)))


#For degree 3 polynomial
deg3 = 3
X3 = getPoweredX(deg3, X_train)

first_part = np.linalg.inv(np.dot(X3.transpose(),X3))
second_part = np.dot(first_part, X3.transpose())
W3 = np.dot(second_part, Y)

points = np.linspace(20,160,200).reshape(200,1)
points = np.append(np.ones((200, 1)), points, axis=1)
points = getPoweredX(deg3,points)

ax3.plot(X,Y, "ro")
ax3.axis([0, 160, 0, 1800])
ax3.get_xaxis().set_visible(False)
ax3.plot(points, np.dot(points, W3))
ax3.set_title("Degree=3")
print("Empirical risk for degree 3 model: ", calculateEmpricalRisk(np.dot(X3,W3)))

#For degree 4 polynomial
deg4 = 4
X4 = getPoweredX(deg4, X_train)

first_part = np.linalg.inv(np.dot(X4.transpose(),X4))
second_part = np.dot(first_part, X4.transpose())
W4 = np.dot(second_part, Y)

points = np.linspace(20,160,200).reshape(200,1)
points = np.append(np.ones((200, 1)), points, axis=1)
points = getPoweredX(deg4,points)

ax4.plot(X,Y, "ro")
ax4.axis([0, 160, 0, 1800])
ax4.get_xaxis().set_visible(False)
ax4.plot(points, np.dot(points, W4))
ax4.set_title("Degree=4")
print("Empirical risk for degree 4 model: ", calculateEmpricalRisk(np.dot(X4,W4)))






#%%
#####################################
######### GRADIENT METHOD ###########
#####################################

import numpy as np
from matplotlib import pyplot as plt

def calculateEmpricalRisk(yh):
    error = sum((Y - yh)**2)
    return error/len(Y)

# With degree2
X_o = np.array([31,33,31,49,53,69,101,99,143,132,109]).reshape(11,1)
X = np.hstack((np.ones((11,1)), X_o, X_o[:]**2))

Y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
Y = Y.reshape(Y.shape[0],1)

w = np.array([[0],[0],[0]])
lr = 1*10**(-9)
m = len(Y)
points_o = np.linspace(20,160,200).reshape(200,1)
points = np.hstack((np.ones((200,1)), points_o,points_o**2))
risk = []
iteration = 30
for i in range(30):
    
    yh = np.dot(X,w)
    risk.append(calculateEmpricalRisk(yh))
    w = w - (1/m) * lr * np.dot(X.T , (yh-Y))
    
    if i % 5 == 0:
        plt.plot(points_o, np.dot(points,w))
        plt.plot(X_o,Y,"ro")
        plt.show()
        plt.close()

plt.plot(range(iteration), risk)
        

#%%
#With degree 3
X_o = np.array([31,33,31,49,53,69,101,99,143,132,109]).reshape(11,1)
X = np.hstack((np.ones((11,1)), X_o, X_o[:]**2,X_o[:]**3))

Y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
Y = Y.reshape(Y.shape[0],1)

w = np.array([[100],[18],[-0.1],[-0.00012]])
lr = 1*10**(-12)
m = len(Y)
points_o = np.linspace(20,160,200).reshape(200,1)
points = np.hstack((np.ones((200,1)), points_o,points_o**2, points_o**3))
risk = []
iteration = 20
for i in range(iteration):
    
    yh = np.dot(X,w)
    risk.append(calculateEmpricalRisk(yh))
    w = w - (1/m) * lr * np.dot(X.T , (yh-Y))
    if i % 4 == 0:
        plt.plot(points_o, np.dot(points,w))
        plt.plot(X_o,Y,"ro")
        plt.show()
        plt.close()
        
plt.plot(range(iteration), risk)

#%%
#with degree 4
X_o = np.array([31,33,31,49,53,69,101,99,143,132,109]).reshape(11,1)
X = np.hstack((np.ones((11,1)), X_o, X_o[:]**2, X_o[:]**3, X_o[:]**4))

Y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
Y = Y.reshape(Y.shape[0],1)

w = np.array([[0],[15.2368],[-0.134594],[0],[0]])
lr = 1*10**(-17)
m = len(Y)
points_o = np.linspace(20,160,200).reshape(200,1)
points = np.hstack((np.ones((200,1)), points_o,points_o**2,points_o**3,points_o**4))
risk = []
iteration = 15
for i in range(15):
  
    yh = np.dot(X,w)
    
    risk.append(calculateEmpricalRisk(yh))
    w = w - (1/m) * lr * np.dot(X.T , (yh-Y))
    if i % 2 == 0:
        plt.plot(points_o, np.dot(points,w))
        plt.plot(X_o,Y,"ro")
        plt.show()
        plt.close()

plt.plot(range(iteration), risk)










