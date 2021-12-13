import numpy as np
import os
from Utils.Common import *
import copy

def com(data):
    x = copy.copy(data)
    x = x[~np.isnan(x).any(axis=1)]
    return np.sum(x,axis=0)/len(x)

def CrossProduct2d(v1,v2):
    return v1[0]*v2[1] - v2[0]*v1[1]

def getRealStartData(path2Data):
    
    traj_data = np.load(path2Data,allow_pickle=True).item(0)
    trajectorie_data = traj_data["trajectories"]
    return trajectorie_data
p_realdata =  "/mnt/HDD-1/schristo/CollectiveBehaviour/Data/zebrafish_trajectories/80/1/trajectories_wo_gaps.npy"
realdata = getRealStartData(p_realdata)

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

def getDynamicalStates(path):
    newpath = path.replace(path.split("/")[-1],"")
    filename = "Rotation"
    path2file = os.path.join(newpath,filename)
    try:
        data = np.load(path2file+".npz",allow_pickle=True)
        Op,Or=data["Op"],data["Or"]
        print("Found it")
    except:
        print("Didnt find it")
        data = getRealStartData(path)
        Op,Or = DynamicalStates(data)
        np.savez(path2file,Op=Op,Or=Or)
    return Op,Or


def PlotParameterWeighted(boids,filename):


    Op,Or = DynamicalStates(realdata)

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter
    
    minerror_omega0 = [i[0][0] for i in boids[0][2:]] 
    minerror_omega1 = [i[0][1] for i in boids[0][2:]] 
    minerror_omega2 = [i[0][2] for i in boids[0][2:]]
    minerror_error  = [i[1] for i in boids[0][2:]]

    minerror_d_omega0 = [i[0][0] for i in boids[1][2:]] 
    minerror_d_omega1 = [i[0][1] for i in boids[1][2:]] 
    minerror_d_omega2 = [i[0][2] for i in boids[1][2:]] 
    minerror_d_error  = [i[1] for i in boids[1][2:]]
    
    minerror_ds_omega0 = [i[0][0] for i in boids[2][2:]] 
    minerror_ds_omega1 = [i[0][1] for i in boids[2][2:]] 
    minerror_ds_omega2 = [i[0][2] for i in boids[2][2:]] 
    minerror_ds_error  = [i[1] for i in boids[2][2:]] 
    
    x = np.arange(0,len(minerror_omega0))

    fig, axs = plt.subplots(5, 1, tight_layout=True,figsize=(30,15))
    axs[0].plot(x,minerror_omega0,color="red")
    #axs[0].plot(x,minerror_d_omega0,color="blue")
    #axs[0].plot(x,minerror_ds_omega0,color="green")
    
    axs[1].plot(x,minerror_omega1,color="red")
    #axs[1].plot(x,minerror_d_omega1,color="blue")
    #axs[1].plot(x,minerror_ds_omega1,color="green")
    
    axs[2].plot(x,minerror_omega2,color="red")
    #axs[2].plot(x,minerror_d_omega2,color="blue")
    #axs[2].plot(x,minerror_ds_omega2,color="green")
    
    axs[3].plot(x,minerror_error,color="red")
    #axs[3].plot(x,minerror_d_error,color="blue")
    #axs[3].plot(x,minerror_ds_error,color="green")
    
    axs[4].plot(Or[2:len(minerror_error)+2],color="red",label="Rotation Or")
    axs[4].plot(Op[2:len(minerror_error)+2],color="blue",label="Polarization Op")

    # We can also normalize our inputs by the total number of counts
    axs[4].set_ylabel("Order of Polarization/Rotation")
    axs[4].set_xlabel("Frame")


    #axs.plot(error_NEG,color="blue",label="Parameter {} = {:.2f}".format(parameterOff[0],parameterOff[2]))
    l = ["Omega 1","Omega 2","Omega 3","l2-loss"]
    for i in range(len(l)):
        axs[i].set_ylabel(l[i])
        axs[i].set_xlabel("Frame")

    plt.legend()
    fig.savefig(filename, dpi=fig.dpi)

def PlotParameterBoids(boids,filename):


    Op,Or = DynamicalStates(realdata)

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter
    
    minerror_R = [i[0][0] for i in boids[0][2:]] 
    minerror_O = [i[0][1] for i in boids[0][2:]] 
    minerror_A = [i[0][2] for i in boids[0][2:]]
    minerror_V = [i[0][3] for i in boids[0][2:]]
    minerror_Alpha = [i[0][4] for i in boids[0][2:]]
    minerror_error  = [i[1] for i in boids[0][2:]]

    minerror_d_R = [i[0][0] for i in boids[1][2:]] 
    minerror_d_O = [i[0][1] for i in boids[1][2:]] 
    minerror_d_A = [i[0][2] for i in boids[1][2:]]
    minerror_d_V = [i[0][3] for i in boids[1][2:]]
    minerror_d_Alpha = [i[0][4] for i in boids[1][2:]]
    minerror_d_error  = [i[1] for i in boids[1][2:]]
    
    minerror_dS_R = [i[0][0] for i in boids[2][2:]] 
    minerror_dS_O = [i[0][1] for i in boids[2][2:]] 
    minerror_dS_A = [i[0][2] for i in boids[2][2:]]
    minerror_dS_V = [i[0][3] for i in boids[2][2:]]
    minerror_dS_Alpha = [i[0][4] for i in boids[2][2:]]
    minerror_ds_error  = [i[1] for i in boids[2][2:]] 
        
    x = np.arange(0,len(minerror_R))

    fig, axs = plt.subplots(7, 1, tight_layout=True,figsize=(30,15))
    axs[0].plot(x,minerror_R,color="red")

    
    axs[1].plot(x,minerror_O,color="red")

    axs[2].plot(x,minerror_A,color="red")

    axs[3].plot(x,minerror_V,color="red")
    
    axs[4].plot(x,minerror_Alpha,color="red")

    axs[5].plot(x,minerror_error,color="red")

    axs[6].plot(Or[2:len(minerror_error)+2],color="red",label="Rotation Or")
    axs[6].plot(Op[2:len(minerror_error)+2],color="blue",label="Polarization Op")

    # We can also normalize our inputs by the total number of counts
    axs[6].set_ylabel("Order of Polarization/Rotation")
    axs[6].set_xlabel("Frame")


    #axs.plot(error_NEG,color="blue",label="Parameter {} = {:.2f}".format(parameterOff[0],parameterOff[2]))
    l = ["R","O","A","V","Alpha","l2-loss"]
    for i in range(len(l)):
        axs[i].set_ylabel(l[i])
        axs[i].set_xlabel("Frame")

    plt.legend()
    fig.savefig(filename, dpi=fig.dpi)


