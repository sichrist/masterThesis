import sys
sys.path.append("../")
from Modelle.PositiveWeightedDistances import Positive_weighted_distances
from Modelle.Boids import BoidsTopologisch,BoidsMetrisch
from SimulateFakeData import GenerateFakeData
from ParticleSwarmOptimization import PSO
from functools import partial
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from Utils.errorFunctions import DynamicalState_error, l2error, l2error_torch, errorOP_Sigmoid,SigmoidError,errorOR_Sigmoid
from datetime import datetime
import os
np.random.seed(150)
from multiprocessing import Process
from time import time
from Utils.Common import getRealData
from Utils.errorFunctions import *
from sys import getsizeof
import gc
def data_generator(data,start_idx=0,end_idx=0):
    if isinstance(data,GenerateFakeData):
        pos_last = data()
        pos_atm  = data()
        pos_true = data()
        yield pos_last,pos_atm,pos_true

        while True:
            pos_last = pos_atm
            pos_atm  = pos_true
            pos_true = data()
            yield pos_last,pos_atm,pos_true

    else:
        if end_idx == 0:
            end = len(data)-1

        pos_last = data[start_idx]
        pos_atm  = data[start_idx+1]
        pos_true = data[start_idx+2]
        yield pos_last,pos_atm,pos_true
        for i in range(start_idx+3,end):
            pos_last = pos_atm
            pos_atm  = pos_true
            pos_true = data[i]
            yield pos_last,pos_atm,pos_true


class PSO_Approximation(object):
    """docstring for PSO_Approximation"""
    def __init__(self, function,errorFunction,filename_suffix="",filename_prefix=""):
        super(PSO_Approximation, self).__init__()
        self.filename_suffix = filename_suffix 
        now = datetime.now() 
        self.default_path = str(self)
        if not os.path.exists(self.default_path):
            os.makedirs(self.default_path)
        self.filename = filename_prefix+"_"+str(function)+"_"+errorFunction.__name__+"_"+filename_suffix
        self.date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.filename += self.date_time
        self.errorFunction = errorFunction
        self.function = function


    def __str__(self):
        return "PSO_Approximation"

    def runOnData(self,data,max_iterations,tol,grid,reset=False,max_frames=np.inf,ppd=30):
        self.reset = reset
        best_params = []
        best_error  = []
        best_prog   = []
        true_pos    = []
        pred_pos    = []
        start = time()
        S = SigmoidError(max_frames)
        gbest = None
        for i,(pos_l,pos_a,pos_true) in enumerate(data_generator(data)):
            if len(true_pos) == 0:
                true_pos.append(pos_l)
                true_pos.append(pos_a)
                true_pos.append(pos_true)
            else:
                true_pos.append(pos_true)
                pred_pos.append(self.function(pos_last,pos_atm,best_params[-1]))


            if i == max_frames:
                break
            if reset == False and i > 0:
                print("No RESET")
                pos_last = pos_atm
                pos_atm = self.function(pos_last,pos_atm,best_params[-1])
            else:
                print("RESET")
                pos_atm = pos_a
                pos_last = pos_l

            g = partial(self.errorFunction,

                Y        = pos_true,
                #Y        = S(i),
                pos_atm  = pos_atm,
                pos_last = pos_last,
                function = self.function)

            gbest_obj,gbest,prog   = PSO(g,
                        self.function.x(),
                        grid,
                        max_iterations=max_iterations,
                        particlesPerDim=ppd,
                        V=0.5,
                        tol=tol,
                        verbose=True,
                        no_change=30,
                        progressPerIter = True,
                        sSPtgB = gbest)
            print("\n{} | {} [{}]".format(i,max_frames,gbest))
            best_params.append(gbest)
            best_error.append(gbest_obj)
            best_prog.append(prog)

        best_params = np.array(best_params,dtype=object)
        best_error = np.array(best_error,dtype=object)
        best_prog = np.array(best_prog,dtype=object)
        true_params=None
        nbr_fish = None
        if isinstance(data,GenerateFakeData):
            true_params=np.array(data.getParams())
            nbr_fish = data.nbr_fish

        np.savez(os.path.join(self.default_path,self.filename),
            best_params=best_params,
            best_error=best_error,
            best_prog=best_prog,
            ppd=ppd,
            nbr_fish = nbr_fish,
            max_frames = max_frames,
            reset = reset,
            true_params=true_params,
            true_pos = np.array(true_pos),
            pred_pos = np.array(pred_pos),
            timespend=start-time())


class RMD_Approximation(object):
    """docstring for RMD_Approximation"""
    def __init__(self, function,errorFunction,filename_suffix="",filename_prefix=""):
        super(RMD_Approximation, self).__init__()
        self.filename_suffix = filename_suffix 
        now = datetime.now() 
        self.default_path = str(self)
        if not os.path.exists(self.default_path):
            os.makedirs(self.default_path)
        self.filename = filename_prefix+"_"+str(function)+"_"+errorFunction.__name__+"_"+filename_suffix
        self.date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.filename += self.date_time
        self.errorFunction = errorFunction
        self.function = function

    def __str__(self):
        return "RMD_Approximation"

    def rmd_(self):
        pass
    def runOnData(self,data,max_iterations,tol,grid,reset=False,max_frames=np.inf,lr=1e-2,epochs=1000):
        self.reset = reset
        best_params = []
        best_error  = []
        best_prog   = []
        true_pos    = []
        pred_pos    = []
        start = time()
        dtype=torch.DoubleTensor
        for i,(pos_l,pos_a,pos_true) in enumerate(data_generator(data)):

            if len(true_pos) == 0:
                true_pos.append(pos_l)
                true_pos.append(pos_a)
                true_pos.append(pos_true)
            else:

                true_pos.append(np.array(pos_true,dtype=np.float32))

                pred_pos.append(self.function(pos_atm,pos_last,best_params[-1]))
                print(type(pred_pos[-1]))
            if i == max_frames:
                break
            if reset == True and i > 0:
                pos_last = pos_atm
                pos_atm = self.function(pos_atm,pos_last,best_params[-1])

            else:
                pos_atm = pos_a
                pos_last = pos_l
            distance = None
            while distance is None:
                cost = np.inf
                par  = None
                x0 = np.array([np.random.uniform(l,h) for l,h in grid])

                kosten = []
                x = Variable(torch.tensor([x0]).type(dtype), requires_grad=True)
                optimizer = optim.Adam([x], lr=lr)
                for j in range(epochs):
                    optimizer.zero_grad()
                    distance = self.errorFunction(x,pos_true,pos_atm,pos_last,self.function)
                    distance.backward(retain_graph=True)
                    optimizer.step()
                    print("Frame: {:4d} | {} - {:.11f}    {}".format(i,j,distance,*x.data.detach().numpy())+" "*0,end="\r") 
                    if distance < cost:
                        par = x.data.detach().numpy().astype(np.float32)
                        cost = distance
                    kosten.append([j,distance.type(torch.float32)])
            print("\nFrame: {:4d} |  {:.11f}".format(epochs,cost)+" "*60,end="\r") 


            del optimizer
            del x
            del distance
            best_params.append(par[0])
            best_error.append(cost.detach().numpy().astype(np.float32))
            best_prog.append(kosten)
            print()
            print(getsizeof(best_params))
            print(getsizeof(best_error))
            print(getsizeof(best_prog))
            print()
            del par
            del kosten
            gc.collect()


        best_params = np.array(best_params,dtype=object).astype(np.float32)
        best_error = np.array(best_error,dtype=object).astype(np.float32)
        best_prog = np.array(best_prog,dtype=object).astype(np.float32)
        true_params=None
        nbr_fish = None
        if isinstance(data,GenerateFakeData):
            true_params=np.array(data.getParams())
            nbr_fish = data.nbr_fish

        np.savez(os.path.join(self.default_path,self.filename),
            best_params=best_params,
            best_error=best_error,
            best_prog=best_prog,
            nbr_fish = nbr_fish,
            max_frames = max_frames,
            reset = reset,
            true_params=true_params,
            true_pos = np.array(true_pos),
            pred_pos = np.array(pred_pos),
            timespend=start-time())


def BoidsM(max_iterations,tol,data,max_frames,reset=False,filename_prefix=""):
    function = BoidsMetrisch()
    randomRange=[(0,200),(0,300),(0,400),(1,90),(0,360)]
    #randomRange=[(0,1000),(0,1000),(0,1000),(0,360),(0,360)]


    pso_instanz = PSO_Approximation(function=function,
                                    errorFunction=l2error,
                                    #errorFunction=DynamicalState_error,
                                    #errorFunction=errorOP_Sigmoid,
                                    #errorFunction=errorOR_Sigmoid,
                                    filename_suffix="_RS_"+str(reset),
                                    filename_prefix=filename_prefix)


    pso_instanz.runOnData(data,max_iterations,tol,randomRange,max_frames=max_frames,reset=reset)

def PSO_realdata(path2file,filename_prefix):
    tol=1e-15
    max_iterations=200
    max_frames=2000
    pidlist = []
    data = getRealData(path2file)
    BoidsM(max_iterations,tol,data,max_frames,reset=False,filename_prefix=filename_prefix)

def RMD(path2file,filename_prefix):
    randomRange = True
    randomParameter = False
    max_iterations=5000
    max_frames = 1000
    reset = False
    tol=1e-15
    function = Positive_weighted_distances()

    randomRange=[(10.0,10.0),(1.0,1.0),(1.0,1.0),(1.0,1.0)]
    data = getRealData(path2file)

    rmd_instanz = RMD_Approximation(function=function,
                                    errorFunction=l2error_torch,
                                    filename_suffix="_RS_"+str(reset),
                                    filename_prefix=filename_prefix)

    randomRange=[(0.0,10.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)]
    rmd_instanz.runOnData(data,max_iterations,tol,randomRange,max_frames=max_frames,reset=reset)

def main():
    #path2realData = "/mnt/HDD-1/schristo/CollectiveBehaviour/Data/zebrafish_trajectories/10/1/trajectories_wo_gaps.npy"
    path2realData = "/mnt/HDD-1/schristo/CollectiveBehaviour/Data/zebrafish_trajectories/60/1/trajectories_wo_gaps.npy"

    #PSO_realdata(path2realData,filename_prefix="zebrafish_trajectories_60_1")
    RMD(path2realData,filename_prefix="zebrafish_trajectories_60_1")



if __name__ == '__main__':
    main()