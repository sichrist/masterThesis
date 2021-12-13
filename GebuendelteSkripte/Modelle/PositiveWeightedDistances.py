from Utils.Common import *
import numpy as np
import torch
dtype = torch.cuda.FloatTensor
from Utils.plotParam import DynamicalStates
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
def Positive_weighted_distances_SingleFish(pos_atm,pos_last,gamma,omega1,omega2,omega3,ID):

    pos_last = np.nan_to_num(pos_last)
    pos_atm  = np.nan_to_num(pos_atm)
    beta     = (1/80)
    rad      = 3533/2

    def getNeareast(pos,id):
        f = pos[id]
        distances = [(dist(pos[i],f),i) for i in range(len(pos)) if i != id]
        distances = sorted(distances, key=lambda tup : tup[0])
        return distances[0]
    
    v1 = []
    v2 = []
    v3 = []
    center_of_school = com(pos_atm)
    v2 = np.array(normalize_Vec(np.sum(pos_atm - pos_last,axis=(0))),dtype=np.float32)

    
    d,id   = getNeareast(pos_atm,ID)
    dirc   = direction(pos_atm[ID],pos_atm[id])
    dirR   = direction(center_of_school,pos_atm[id])

    v1.append(normalize_Vec(dirc))
    v3.append(normalize_Vec(dirR))

    facV1 = np.exp(-VecLen(dirc)*beta)
    #facV2 = np.exp(-VecLen(dirc)*beta)
    facV2 = 1.0
    facV3 = np.exp(VecLen(dirR)/rad)-0.8
    v1 = np.array(v1,dtype=np.float32)*facV1
    v3 = np.array(v3,dtype=np.float32)*facV3
    v2 *= facV2
    vec = gamma*(omega1*v1 + omega2*v2 + omega3*v3)

    return pos_atm[ID]+vec

#def Positive_weighted_distances_MultiFish_RMD(pos_atm,pos_last,gamma,omega1,omega2,omega3):
def Positive_weighted_distances_MultiFish_RMD(pos_atm,pos_last,x,dtype=torch.DoubleTensor):
    pos_last = np.nan_to_num(pos_last)
    pos_atm  = np.nan_to_num(pos_atm)
    beta     = (1/80)
    rad      = 3533/2
    distances = np.zeros((len(pos_atm),len(pos_atm)))
    directions = np.zeros((len(pos_atm),*pos_atm.shape))

    for i in range(len(pos_atm)):
        dirc=np.repeat([pos_atm[i]],len(pos_atm),axis=0)-pos_atm
        directions[:,i,:] = dirc
        distances[:,i]=np.sqrt(np.sum(dirc**2,axis=1))
        distances[i,i] = np.inf
    m = np.min(distances,axis=0)
    mi = np.argmin(distances,axis=0)

    ind = np.arange(0,len(mi))

    dist = distances[mi,ind]
    beta     = (1/80)
    rad      = 3533/2
    max_d    = distances[np.isfinite(distances)].mean()/4
    center_of_school = com(pos_atm)
    vec2center = center_of_school - pos_atm
    dist2center = veclen(vec2center)
    v1 = normalize(directions[mi,ind])
    v2 = normalize(pos_atm-pos_last)
    v3 = normalize(vec2center)
    facV1 = np.exp(-dist*beta)
    if max_d == 0:
        max_d = 1
    facV2 = np.exp(-distances/max_d)
    #facV3 = np.exp(-(dist2center/rad))
    facV3 = 1/(1+np.exp(-(dist2center/(rad/4))))

    v1 *= np.expand_dims(facV1,-1)
    v2 *= np.sum(np.expand_dims(facV2,-1),0)
    v3 *= np.expand_dims(facV3,-1)


    if torch.is_tensor(x):
        v1 = torch.tensor(v1).type(dtype)
        v2 = torch.tensor(v2).type(dtype)
        v3 = torch.tensor(v3).type(dtype)
        vec = x[:,0].unsqueeze(1)*(x[:,1].unsqueeze(1)*v1 + x[:,2].unsqueeze(1)*v2 + x[:,3].unsqueeze(1)*v3)
        p = torch.tensor(pos_atm).type(dtype)
        return p+vec

    vec = (x[1]*v1 + x[2]*v2 + x[3]*v3)
    return pos_atm.copy() + vec
    last_direction = pos_atm-pos_last
    #last_direction = last_direction/torch.sqrt(torch.sum(last_direction**2,axis=-1)).unsqueeze(-1)
    for i in range(len(vec)):
        vi = turnmax(last_direction[i],vec[i],x[4])
        if np.isnan(np.rad2deg(angle_between(last_direction[i],vi))):
            vi = last_direction[i]
        vec[i] = vi
    vec = vec/np.expand_dims(np.sqrt(np.sum(vec**2,axis=-1)),-1)
    vec = x[0]*vec
    if np.isnan(vec).any():
        print(vec)

    d = pos_atm.copy() + vec
    return d



def veclen(x):
    return np.sqrt(np.sum(x**2,axis=1))
def normalize(x):
    return np.divide(x,veclen(x).reshape(len(x),1))

def PWD_RMD (pos_atm,pos_last,x):

    #gamma,omega1,omega2,omega3 = x
    pos_last = np.nan_to_num(pos_last)
    pos_atm  = np.nan_to_num(pos_atm)
    beta     = (1/80)
    rad      = 3533/2

    def getNeareast(pos,id):
        f = pos[id]
        distances = [(dist(pos[i],f),i) for i in range(len(pos)) if i != id]
        distances = sorted(distances, key=lambda tup : tup[0])
        return distances[0]
    
    v1 = []
    v2 = []
    v3 = []
    center_of_school = com(pos_atm)
    v2 = np.array(pos_atm - pos_last,dtype=np.float32)
    v2 = normalize(v2)
    v3 = normalize(center_of_school-pos_atm)
    v3 = v3*np.expand_dims(np.exp(veclen(center_of_school-pos_atm)/rad)-0.8,-1)
    
    ds = []
    for i in range(len(pos_atm)):
        dirc = direction(pos_atm[i],pos_atm[getNeareast(pos_atm,i)[1]])
        v1.append(np.array(dirc)*np.exp(-VecLen(dirc)*beta))


    v1 = torch.tensor(v1).type(dtype)
    v2 = torch.tensor(v2).type(dtype)
    v3 = torch.tensor(v3).type(dtype)

    vec = x[:,0].unsqueeze(1)*(x[:,1].unsqueeze(1)*v1 + x[:,2].unsqueeze(1)*v2 + x[:,3].unsqueeze(1)*v3)
    p = torch.tensor(pos_atm).type(dtype)
    return p+vec

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class Positive_weighted_distances(object):
    """docstring for Positive_weighted_distances"""
    def __init__(self,grid=None):
        super(Positive_weighted_distances, self).__init__()
        self.grid = grid
        self.pixelToCM = 0.16

    def ParameterNames(self):
        return ["Gamma","Repulsion Alpha","Orientation Alpha","Atraction Alpha"]

    def plotIt(self,data,path2file,pred_pos = None):
            #os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
            parameter = data["best_params"]
            pred_data = data["pred_pos"]
            if pred_pos is not None:
                pred_data = pred_pos
            true_data = data["true_pos"]
            R = parameter[:0]
            O = parameter[:1]
            A = parameter[:2]

            Op_pred,Or_pred= DynamicalStates(pred_data)
            Op_true,Or_true = DynamicalStates(true_data)
            sigma = 1
            fig, axs = plt.subplots(parameter.shape[-1]+3, 1,figsize=(20,40))

            pixelToCM = 1.0
            pixelToCM = self.pixelToCM
            toTime = 32
            x = np.arange(len(Op_true))
            axs[0].plot(np.arange(len(Op_pred))/toTime,gaussian_filter1d(Op_pred,sigma),label="Op pred",color="red",linewidth=3)
            axs[1].plot(np.arange(len(Or_pred))/toTime,gaussian_filter1d(Or_pred,sigma),label="Or pred",color="blue",linewidth=3)
            axs[1].plot(x/toTime,gaussian_filter1d(Or_true,sigma),label="Or True",color="blue",alpha=0.4,linewidth=4)
            axs[0].plot(x/toTime,gaussian_filter1d(Op_true,sigma),label="Op True",color="red",alpha=0.4,linewidth=4)
            axs[0].set_ylabel("Rotation/Polarization",fontsize=18)
            axs[0].tick_params(axis='both', which='major', labelsize=18)
            axs[1].set_ylabel("Rotation/Polarization",fontsize=18)
            axs[1].tick_params(axis='both', which='major', labelsize=18)
            label = self.ParameterNames()
            for i in range(1,parameter.shape[-1]+1):
                if "true_params" in data:

                    true_params = data["true_params"]

                    try:
                        axs[i+1].plot(np.arange(len(true_params[:,i-1]))/toTime,
                            gaussian_filter1d(true_params[:,i-1].astype(np.float)*pixelToCM,sigma),
                            label = label[i-1]+" True",alpha=0.9,linewidth=3,color="orange")
                        axs[i+1].tick_params(axis='both', which='major', labelsize=18)
                    except:
                        pass
                axs[i+1].plot(np.arange(len(parameter[:,i-1]))/toTime,
                    gaussian_filter1d(parameter[:,i-1].astype(np.float)*pixelToCM,sigma),
                    label = label[i-1]+" Predicted",linewidth=3,color="blue")
                axs[i+1].set_ylabel("Gewichtung",fontsize=20)
                axs[i+1].tick_params(axis='both', which='major', labelsize=18)

           
            axs[-1].plot(np.arange(len(data["best_error"])),data["best_error"],label="l2 error",linewidth=3)
            for i in range(len(axs)):
                axs[i].legend(loc="upper right",prop={'size': 18})
                axs[i].set_xlabel("Zeit in S",fontsize=20)
                axs[i].grid(True)
                axs[i].tick_params(axis='both', which='major', labelsize=18)
                axs[i].tick_params(axis='both', which='major', labelsize=18)
                axs[i].tick_params(axis='both', which='major', labelsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
            axs[-1].set_xlabel("Sequenz",fontsize=20)
            axs[-1].set_ylabel("l2-error",fontsize=20)

            path = path2file.replace(".npz","")
            if not os.path.exists(path):
                os.makedirs(path)


            plt.savefig(os.path.join(path,"parameter.png"))

    def x(self):
        if self.grid is None:
            return np.abs(np.random.uniform(size=(1,4)))
        x0 = np.array([np.random.uniform(l,h) for l,h in self.grid])
        return 
    def __str__(self):
        return "Positive_weighted_distances"

    def __call__(self,pos_atm,pos_last,x,dtype=torch.DoubleTensor,printit=False):
        pixelToCM = self.pixelToCM

        pos_last = np.nan_to_num(pos_last)
        pos_atm  = np.nan_to_num(pos_atm)
        """
        """
        #beta     = (1/80)
        beta     = 1/3
        rad      = 3533/2
        distances = np.zeros((len(pos_atm),len(pos_atm)))
        directions = np.zeros((len(pos_atm),*pos_atm.shape))

        for i in range(len(pos_atm)):
            dirc=np.repeat([pos_atm[i]],len(pos_atm),axis=0)-pos_atm
            directions[:,i,:] = dirc
            distances[:,i]=np.sqrt(np.sum(dirc**2,axis=1))
            distances[i,i] = np.inf
        m = np.min(distances,axis=0)
        mi = np.argmin(distances,axis=0)

        ind = np.arange(0,len(mi))

        dist = distances[mi,ind]
        #beta     = (1/150)
        #rad      = 3533/2
        #max_d    = distances[np.isfinite(distances)].mean()/4
        center_of_school = com(pos_atm)
        vec2center = center_of_school - pos_atm
        dist2center = veclen(vec2center)
 


        v1 = np.nan_to_num(normalize(directions[mi,ind]))
        facV1 = np.exp((-distances[mi,ind]/20)*beta)
        v1 *= np.expand_dims(facV1,-1)


        v2 = np.nan_to_num(normalize(pos_atm-pos_last))
        v3 = np.nan_to_num(normalize(vec2center))

        # Alte Faktoren
        #if max_d == 0:
        #    max_d = 1

        """
        facV3 = np.exp(-(dist2center/rad))
        facV3 = 1/(1+np.exp(-(dist2center/(rad/4))))
        #facV2 = gaussian(distances,300,100)
        facV2 = gaussian(dist,1000,300)
        x = (-dist+(2000))/8000
        facV3 = 1 / (1 + np.exp(x/pixelToCM))
        facV1 = np.exp(-dist*beta*pixelToCM)
        facV3 = 1/(1+np.exp(-(dist2center/(rad/4))))
        """
        #facV1 = np.exp((-distances/20)*beta)

        #v1 *= np.expand_dims(facV1,-1)
        #v2 *= np.sum(np.expand_dims(facV2,-1),0)
        #v3 *= np.expand_dims(facV3,-1)

        facV1 = 1 / (1 + np.exp((dist-(10/pixelToCM))))
        facV2 = gaussian(dist,15/pixelToCM,3/pixelToCM)
        facV3 = 1 / (1 + np.exp(-(dist-(25/pixelToCM))/(5/pixelToCM)))


        v1 *= np.expand_dims(facV1,-1)
        v2 *= np.sum(np.expand_dims(facV2,-1),0)
        v3 *= np.expand_dims(facV3,-1)

        if torch.is_tensor(x):
            v1 = torch.tensor(v1, dtype=torch.float64)
            v2 = torch.tensor(v2, dtype=torch.float64)
            v3 = torch.tensor(v3, dtype=torch.float64)
            vec = x[:,0].unsqueeze(1)*(x[:,1].unsqueeze(1)*v1 + x[:,2].unsqueeze(1)*v2 + x[:,3].unsqueeze(1)*v3)
            p = torch.tensor(pos_atm).type(dtype)
            if printit:
                print(vec)
                print(".....")
                print(">> ",x[:,0]*(x[:,1]+x[:,2]+x[:,3]))
            return p+vec

        vec = x[0]*(x[1]*v1 + x[2]*v2 + x[3]*v3)
        if printit:
            print(vec)
            print("....")
            print(x[0]*(x[1] + x[2] + x[3]))
        d = pos_atm.copy() + vec
        return d

        