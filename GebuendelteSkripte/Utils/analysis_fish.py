import copy
import numpy as np
from Utils.Common import *
def com(data):
    x = copy.copy(data)
    x = x[~np.isnan(x).any(axis=1)]
    return np.sum(x,axis=0)/len(x)

def CrossProduct2d(v1,v2):
    return v1[0]*v2[1] - v2[0]*v1[1]

def DynamicalStates(data):
    #Order of alignment 0 -> no alignment, 1 -> Strong alignment
    Op = []
    #Order of Rotation 0 -> no rotation, 1 -> Strong rotation
    Or = []
    for i in range(1,len(data)):
        sumOfDire = np.array([0.0,0.0])
        sumOfRot  = 0.0
        nbr = 0
        CoM = com(data[i])
        for (x0,y0),(x1,y1) in zip(data[i-1],data[i]):
            if np.isnan(x0) or np.isnan(x1) or np.isnan(y0) or np.isnan(y1):
                continue
            
            ui = normalize_Vec(direction([x0,y0],[x1,y1]))
            ri = normalize_Vec(direction([x1,y1],CoM)) 

            sumOfRot += CrossProduct2d(ui,ri)
            dire = normalize_Vec(ui)
            sumOfDire+=dire
            nbr += 1
        Op.append(VecLen(sumOfDire)/nbr)
        Or.append(np.abs(sumOfRot/nbr))
    return Op,Or