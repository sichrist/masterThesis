"""
        File for Particel Swarm Optimization (PSO)
"""

import numpy as np
import time
import sys
from Utils.Common import bcolors
def ackleys(x1,x2):
        return -20.0*np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2)))-np.exp(0.5*(np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2)))+20+np.e

def Rastrigin(x,A=10):
    n = 1
    x = np.array([x]).flatten()

    if hasattr(x,'__len__'):
        n = len(x)

    return A*n + np.sum(x**2-A*np.cos(2*np.pi*x))



def PSO_Multi(f,
    grid,
    particlesPerDim=20,
    max_iterations=1000,
    V=0.1,
    tol=1e-50,
    verbose=True,
    Queue=None,
    info=None,
    c1  = 0.1,
    c2  = 0.1,
    w   = 0.8,
    no_change=10,
    sSPtgB = None):

    pass
    print(grid)
    X = np.random.uniform(0,1,).reshape(x.shape)


#V = X = pbest = pbest_obj =  gbest =  gbest_obj = 0

def PSO(f,x,
    grid,
    particlesPerDim=20,
    max_iterations=1000,
    V=0.1,
    tol=1e-50,
    verbose=True,
    Queue=None,
    info=None,
    c1  = 0.1,
    c2  = 0.1,
    w   = 0.8,
    no_change=10, 
    progressPerIter=False,
    sSPtgB = None):
    np.random.seed(None)
    #global V, X, pbest, pbest_obj, gbest, gbest_obj
    if hasattr(x,'__len__'): 
        nbr_particles = particlesPerDim*len(x)
    else:
        nbr_particles = particlesPerDim
    X = np.abs(np.array([np.random.uniform(low=l,high=h,size=nbr_particles) for l,h in grid]))
    V = np.array([np.random.uniform(low=-1.0,high=1.0,size=nbr_particles) for i in grid]) * V

    pbest = X
    if sSPtgB is not None:
        """
            Set one pbest to a specific parameter 
            in case we know where the global best (approximatly) should be
        """

        pbest[:,0] = sSPtgB

    pbest_obj = np.array([f(u) for u in X.transpose()])
    gbest = pbest[:,pbest_obj.argmin()]
    gbest_obj = pbest.min()
    c1 = c2 = 2
    w = 0.8
    last_change = 0
    last_best   = 0
    last_obj    = gbest_obj
    color = bcolors.OKGREEN
    fill = len(str(max_iterations))

    if progressPerIter == True:
        prog = []
    for i in range(max_iterations):


        if verbose:
            if last_change >= no_change//2:
                color = bcolors.WARNING
            elif last_change >= no_change:
                color = bcolors.FAIL
            elif last_change == 1:
                color = bcolors.BOLD
            else:
                color = bcolors.OKGREEN
            if np.abs(gbest_obj) <= tol:
                color = bcolors.OKCYAN 
            if i >= max_iterations//4:
                color_iter = bcolors.FAIL
            else:
                color_iter = bcolors.OKCYAN
            d = np.abs(tol)/np.abs(gbest_obj)

            if d > 0.5:
                color_dist = bcolors.OKBLUE
            elif d > 0.25:
                color_dist = bcolors.OKCYAN
            else:
                color_dist = bcolors.RED
            out = "{}{}{} - tol: {}{:.1e}{}   {:.1e}  Ratio: {}{:.1e}{}         ".format(color_iter,str(i).zfill(fill),bcolors.ENDC,color,gbest_obj,bcolors.ENDC,tol,color_dist,d,bcolors.ENDC)
            if Queue is not None:
                try:
                    Queue.put((info,out),block=False)
                except:
                    pass
            else:
                print(out,end="\r")




        r1,r2 = np.random.rand(2)


        #r1 = np.random.uniform(0,1,X.shape)
        #r2 = np.random.uniform(0,1,X.shape)
        #print((pbest - X).shape,r.shape)
        if gbest_obj == last_obj:

            last_change+=1

        else:

            last_change = 0 

            last_obj = gbest_obj.copy()

        if last_change == no_change:

            break
        if np.abs(gbest_obj) < tol:

            break
        V = w*V+c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
        X = np.abs(X + V)

        obj = np.array([f(u) for u in X.transpose()])


        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj,obj]).min(axis=0)
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()
        if progressPerIter == True:
            prog.append((i,gbest_obj))
    if progressPerIter == True:
        return gbest_obj,gbest,prog
    return gbest_obj,gbest



def main():
    gbest_obj,gbest = PSO(Rastrigin,[20,20,20,2],
                        max_iterations=1000,
                        grid=[(-50,50),(-50,50),(-50,50),(-50,50)],
                        particlesPerDim=1000,
                        no_change=1000,
                        tol=1e-15)

    PSO_Multi(Rastrigin,
                        max_iterations=1000,
                        grid=[(-50,50),(-50,50),(-50,50),(-50,50)],
                        particlesPerDim=1000,
                        no_change=1000,
                        tol=1e-15)


if __name__ == '__main__':
        main()