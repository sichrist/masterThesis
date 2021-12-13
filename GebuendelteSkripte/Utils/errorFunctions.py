import torch
import numpy as np
from torch.autograd import Variable


        

def cross2D(arr1,arr2):
    return arr1[:,0]*arr2[:,1] - arr2[:,0]*arr1[:,1]

def veclen_torch(arr):

    arr = torch.sqrt(torch.sum(torch.pow(arr,2),axis=1))
    return arr.unsqueeze(1)


def normalize_torch(arr):
    return torch.nan_to_num(arr/veclen_torch(arr))

def CrossProduct2d(v1,v2):
    return v1[0]*v2[1] - v2[0]*v1[1]\

def getRotPol(p1,p2):
    com = torch.sum(p1,axis=-0)/len(p1)

    direction2center = com - p1
    ri = normalize_torch(direction2center)
    ui = normalize_torch(p1-torch.tensor(p2))

    mean_dir = torch.mean(ui,axis=0)


    degreeOrPol      = torch.sqrt(torch.sum((mean_dir**2)))
    degreeOfRotation = cross2D(ui,ri)
    degreeOfRotation_norm = torch.abs(torch.sum(degreeOfRotation)/len(degreeOfRotation))
    return degreeOrPol,degreeOfRotation_norm

def DynamicalState_error(X,Y,pos_atm,pos_last,function,param=False):

    pos_pred = function(pos_atm,pos_last,X)
    Op_T,Or_T = getRotPol(torch.tensor(Y),torch.tensor(pos_atm))
    Op_P,Or_P = getRotPol(torch.tensor(pos_pred),torch.tensor(pos_atm))

    if param:
        return Op_T,Or_T,Op_P,Or_P
    if Or_T > Op_T:
        return np.abs(Or_T-Or_P)
    else:
        return np.abs(Op_T-Op_P)
    #return Op_T**2-Op_P**2 + Or_T**2-Or_P**2

def l2error(X,Y,pos_atm,pos_last,function):

    pos_atm = np.nan_to_num(pos_atm)
    pos_last = np.nan_to_num(pos_last)
    prediction = function(pos_atm,pos_last,X)
    if torch.is_tensor( prediction):
        prediction = prediction.detach().numpy()
    if torch.is_tensor(Y):
        Y = Y.detach().numpy()

    #return np.sum(np.abs(Y-prediction))
    
    return np.sum((Y-prediction)**2)

 
def l2error_torch(X,Y,pos_atm,pos_last,function,dtype=torch.DoubleTensor):
    target = Variable(torch.DoubleTensor(Y).type(dtype),requires_grad=False)
    y = function(pos_atm,pos_last,X)
    return torch.sum((torch.pow((y - target), 2)))


class SigmoidError(object):
    """docstring for SigmoidError"""
    def __init__(self, frames):
        super(SigmoidError, self).__init__()
        self.x = np.arange(frames)
        self.frames = frames
        self.__name__=self.__name__()
    def __name__(self):
        return "SigmoidError"
    def sigmoid(self,x):
        return 1 / (1 + np.exp((-x+(self.frames//2))/50))
    def __call__(self,frame):
        y = self.sigmoid(self.x[frame])
        return y,1.0-y


def errorOP_Sigmoid(X,Y,pos_atm,pos_last,function,param=False):
    pos_pred = function(pos_atm,pos_last,X)
    Op_P,Or_P = getRotPol(torch.tensor(pos_pred),torch.tensor(pos_atm))
    Op_T,Or_T = Y

    if param:
        return Op_T,Or_T,Op_P,Or_P
    return np.abs(Op_T-Op_P)

def errorOR_Sigmoid(X,Y,pos_atm,pos_last,function,param=False):
    pos_pred = function(pos_atm,pos_last,X)
    Op_P,Or_P = getRotPol(torch.tensor(pos_pred),torch.tensor(pos_atm))
    Op_T,Or_T = Y

    if param:
        return Op_T,Or_T,Op_P,Or_P
    return np.abs(Or_T-Or_P)
