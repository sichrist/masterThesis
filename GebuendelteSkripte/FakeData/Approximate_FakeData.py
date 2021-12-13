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

        pos_last = data(start_idx)
        pos_atm  = data[start_idx+1]
        pos_true = data(start_idx+2)
        yield pos_last,pos_atm,pos_true
        for i in range(start_idx+3,end):
            pos_last = pos_atm
            pos_atm  = pos_true
            pos_true = data[i]
            yield pos_last,pos_atm,pos_true


class RMD_Approximation(object):
    """docstring for RMD_Approximation"""
    def __init__(self, function,errorFunction,filename_suffix=""):
        super(RMD_Approximation, self).__init__()
        self.filename_suffix = filename_suffix 
        now = datetime.now() 
        self.default_path = str(self)
        if not os.path.exists(self.default_path):
            os.makedirs(self.default_path)
        self.filename = str(function)+"_"+errorFunction.__name__+"_"+filename_suffix
        self.date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.filename += self.date_time
        self.errorFunction = errorFunction
        self.function = function

    def __str__(self):
        return "RMD_Approximation"

    def rmd_(self):
        pass
    def runOnData(self,data,max_iterations,tol,grid,reset=False,max_frames=np.inf,lr=1e-2,epochs=5000):
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
                true_pos.append(pos_true)

                pred_pos.append(self.function(pos_atm,pos_last,best_params[-1]))
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
                        par = x.data.detach().numpy()
                        cost = distance
                    kosten.append([j,distance])
            print("\nFrame: {:4d} |  {:.11f}".format(epochs,cost)+" "*60,end="\r") 
            print(par[0])
            print(data.getParams()[-1])
            print()

            """
            print(pos_true)
            print(self.function(pos_atm,pos_last,x.data,printit=True))
            print("--------")
            print(self.function(pos_atm,pos_last,data.getParams()[-1],printit=True))
            print()
            """
            del optimizer
            del x
            best_params.append(par[0])
            best_error.append(cost)
            best_prog.append(kosten)

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
            nbr_fish = nbr_fish,
            max_frames = max_frames,
            reset = reset,
            true_params=true_params,
            true_pos = np.array(true_pos),
            pred_pos = np.array(pred_pos),
            timespend=start-time())



class PSO_Approximation(object):
    """docstring for PSO_Approximation"""
    def __init__(self, function,errorFunction,filename_suffix=""):
        super(PSO_Approximation, self).__init__()
        self.filename_suffix = filename_suffix 
        now = datetime.now() 
        self.default_path = str(self)
        if not os.path.exists(self.default_path):
            os.makedirs(self.default_path)
        self.filename = str(function)+"_"+errorFunction.__name__+"_"+filename_suffix
        self.date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.filename += self.date_time
        self.errorFunction = errorFunction
        self.function = function


    def __str__(self):
        return "PSO_Approximation"

    def runOnData(self,data,max_iterations,tol,grid,reset=False,max_frames=np.inf,ppd=50):
        self.reset = reset
        best_params = []
        best_error  = []
        best_prog   = []
        true_pos    = []
        pred_pos    = []
        start = time()
        S = SigmoidError(max_frames)

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
            if reset == True and i > 0:
                pos_last = pos_atm
                pos_atm = self.function(pos_last,pos_atm,best_params[-1])
            else:
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
                        V=0.01,
                        tol=tol,
                        verbose=True,
                        no_change=500,
                        progressPerIter = True)

            print()
            print(gbest)
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


def PWD(max_iterations,tol,max_frames,nbr_fish=10,randomParameter=False,reset=False):

    function = Positive_weighted_distances()
    randomRange=[(0.1,0.1),(1,1),(1,1),(1,1)]
    generator = GenerateFakeData(function,
                            randomRange=randomRange,
                            randomParameter=randomParameter,
                            std=5,
                            nbr_fish = 2,
                            show=False)


    randomRange=[(0.0,1.0),(0,1),(0,1),(0,1)]
    pso_instanz = PSO_Approximation(function=function,
                                    errorFunction=l2error,
                                    filename_suffix="_RP_"+str(randomParameter)+"_RS_"+str(reset)+"_NBR_FISH_"+str(nbr_fish))

    pso_instanz.runOnData(generator,max_iterations,tol,randomRange,max_frames=max_frames,reset=reset)

def BoidsM(max_iterations,tol,max_frames,nbr_fish=10,std=50,randomParameter=False,reset=False):
    function = BoidsMetrisch()
    
    randomRange=[(100,200),(200,400),(400,500),(1,10),(90,180)]
    generator = GenerateFakeData(function,
                                randomRange=randomRange,
                                randomParameter=randomParameter,
                                std=std,
                                nbr_fish = nbr_fish,
                                show=False)

    pso_instanz = PSO_Approximation(function=function,
                                    errorFunction=l2error,
                                    #errorFunction=DynamicalState_error,
                                    #errorFunction=errorOP_Sigmoid,
                                    #errorFunction=errorOR_Sigmoid,
                                    filename_suffix="_RP_"+str(randomParameter)+"_RS_"+str(reset)+"_NBR_FISH_"+str(nbr_fish)+"_STD_"+str(std))
    randomRange=[(100,200),(200,400),(400,500),(1,10),(90,180)]
    pso_instanz.runOnData(generator,max_iterations,tol,randomRange,max_frames=max_frames,reset=reset)

def PSOAPP():
    tol=1e-15
    max_iterations=200
    max_frames=200
    pidlist = []
    fish1,std1 = 20,50
    #fish2,std2 = 30,150
    #fish3,std3 = 60,200
    """
    pidlist.append(Process(target = PWD,args=(max_iterations,tol,max_frames,False,False)))
    pidlist.append(Process(target = PWD,args=(max_iterations,tol,max_frames,True,False)))
    pidlist.append(Process(target = PWD,args=(max_iterations,tol,max_frames,False,True)))
    pidlist.append(Process(target = PWD,args=(max_iterations,tol,max_frames,True,True)))
    """

    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish1,std1,False,False)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish1,std1,True,False)))
    """
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish1,std1,True,False)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish1,std1,False,True)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish1,std1,True,True)))

    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish2,std2,False,False)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish2,std2,True,False)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish2,std2,False,True)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish2,std2,True,True)))


    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish3,std3,False,False)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish3,std3,True,False)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish3,std3,False,True)))
    pidlist.append(Process(target = BoidsM,args=(max_iterations,tol,max_frames,fish3,std3,True,True)))
    """
    for p in pidlist:
        p.start()
    for p in pidlist:
        p.join()

def RMD():
    randomRange = True
    randomParameter = True
    max_iterations=5000
    max_frames = 200
    reset = False
    function = Positive_weighted_distances()
    tol = 1e-15
    nbr_fish = 20
    randomRange=[(0.0,10.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)]
    generator = GenerateFakeData(function,
                            randomRange=randomRange,
                            randomParameter=randomParameter,
                            std=50,
                            nbr_fish = nbr_fish,
                            dsize=(1000,1000),
                            show=False)

    rmd_instanz = RMD_Approximation(function=function,
                                    errorFunction=l2error_torch,
                                    filename_suffix="FISH_"+str(nbr_fish)+"_RP_"+str(randomParameter)+"SW_"+str(randomRange[0][0])+"_")

    randomRange=[(0.0,10.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)]
    rmd_instanz.runOnData(generator,max_iterations,tol,randomRange,max_frames=max_frames,reset=reset)

def main():
    RMD()
    #PSOAPP()
    #PWD(2000,1e-50,50,nbr_fish=2,randomParameter=False,reset=False)

if __name__ == '__main__':
    main()