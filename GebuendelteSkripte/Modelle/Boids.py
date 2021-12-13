from Utils.Common import *
import torch
import matplotlib.pyplot as plt
from Utils.Common import *
from Utils.plotParam import DynamicalStates
from scipy.ndimage import gaussian_filter1d
import os
def moveroutine_boids(pos_atm,last_pos,R,O,A,V,alpha):

    newpos = []
    for i,p1 in enumerate(pos_atm):
        
        Rd = []
        Od = []
        Ad = []
        
        for j,p2 in enumerate(pos_atm):
            if i == j:
                continue
            d = dist(p1,p2)

            if d <= R:
                if p2[0] == p1[0] and p2[1] == p1[1]:
                    continue
                Rd.append(normalize_Vec(direction(p2,p1)))
                
            if d > R and d <= O:
                Od.append(normalize_Vec(direction(pos_atm[j],last_pos[j])))
                
            if d > O and d <= A:
                Ad.append(normalize_Vec(direction(p2,p1)))

        if len(Rd) > 0:
            di = -np.sum(Rd,axis=0) 

        else:
            if len(Od) == 0:
                Od = np.array([[0.0,0.0]])
            if len(Ad) == 0:
                Ad = np.array([[0.0,0.0]])

            di = alpha * np.sum(Od,axis=0) + (1 - alpha)*np.sum(Ad,axis=0)

        di = normalize_Vec(di)

        if np.isnan(di[0]) or np.isnan(di[1]):
            di = [0,0]

        newpos.append([p1[0]+V*di[0],
                       p1[1]+V*di[1]])
        
        pos_atm[i] = newpos[-1]

    return np.array(newpos)


def moveroutine_boids(pos_atm,last_pos,R,O,A,V,alpha):

    newpos = []
    print(R,O,A,V,alpha)
    for i,p1 in enumerate(pos_atm):
        
        Rd = []
        Od = []
        Ad = []
        
        for j,p2 in enumerate(pos_atm):
            if i == j:
                continue
            d = dist(p1,p2)

            if d <= R:
                if p2[0] == p1[0] and p2[1] == p1[1]:
                    continue
                Rd.append(normalize_Vec(direction(p2,p1)))
                
            if d > R and d <= O:
                Od.append(normalize_Vec(direction(pos_atm[j],last_pos[j])))
                
            if d > O and d <= A:
                Ad.append(normalize_Vec(direction(p2,p1)))

        if len(Rd) > 0:
            di = -np.sum(Rd,axis=0) 

        else:
            if len(Od) == 0:
                Od = np.array([[0.0,0.0]])
            if len(Ad) == 0:
                Ad = np.array([[0.0,0.0]])

            di = alpha * np.sum(Od,axis=0) + (1 - alpha)*np.sum(Ad,axis=0)

        di = normalize_Vec(di)

        if np.isnan(di[0]) or np.isnan(di[1]):
            di = [0,0]

        newpos.append([p1[0]+V*di[0],
                       p1[1]+V*di[1]])
        
        pos_atm[i] = newpos[-1]

    return np.array(newpos)

def moveroutine_boids(pos_atm,last_pos,x):
    R,O,A,V,alpha = x
    newpos = []
    print(R,O,A,V,alpha)
    for i,p1 in enumerate(pos_atm):
        
        Rd = []
        Od = []
        Ad = []
        
        for j,p2 in enumerate(pos_atm):
            if i == j:
                continue
            d = dist(p1,p2)

            if d <= R:
                if p2[0] == p1[0] and p2[1] == p1[1]:
                    continue
                Rd.append(normalize_Vec(direction(p2,p1)))
                
            if d > R and d <= O:
                Od.append(normalize_Vec(direction(pos_atm[j],last_pos[j])))
                
            if d > O and d <= A:
                Ad.append(normalize_Vec(direction(p2,p1)))

        if len(Rd) > 0:
            di = -np.sum(Rd,axis=0) 

        else:
            if len(Od) == 0:
                Od = np.array([[0.0,0.0]])
            if len(Ad) == 0:
                Ad = np.array([[0.0,0.0]])

            di = alpha * np.sum(Od,axis=0) + (1 - alpha)*np.sum(Ad,axis=0)

        di = normalize_Vec(di)

        if np.isnan(di[0]) or np.isnan(di[1]):
            di = [0,0]

        newpos.append([p1[0]+V*di[0],
                       p1[1]+V*di[1]])
        
        pos_atm[i] = newpos[-1]

    return np.array(newpos)

def moveroutine_boids_SingleFish(pos_atm,last_pos,R,O,A,V,alpha,ID):

    newpos = []

    fish_atm = pos_atm[ID]
    fish_last = last_pos[ID]
    
    Rd = []
    Od = []
    Ad = []


    for j,p2 in enumerate(pos_atm):
        if j == ID:
            continue

        d = dist(fish_atm,p2)

        if d <= R:
            if p2[0] == fish_atm[0] and p2[1] == fish_atm[1]:
                continue
            Rd.append(normalize_Vec(direction(p2,fish_atm)))
            
        if d > R and d <= O:
            Od.append(normalize_Vec(direction(pos_atm[j],last_pos[j])))
            
        if d > O and d <= A:
            Ad.append(normalize_Vec(direction(p2,fish_atm)))

    if len(Rd) > 0:
        di = -np.sum(Rd,axis=0) 

    else:
        if len(Od) == 0:
            Od = np.array([[0.0,0.0]])
        if len(Ad) == 0:
            Ad = np.array([[0.0,0.0]])

        di = alpha * np.sum(Od,axis=0) + (1 - alpha)*np.sum(Ad,axis=0)

    di = normalize_Vec(di)

    if np.isnan(di[0]) or np.isnan(di[1]):
        di = [0,0]

    newpos = [fish_atm[0]+V*di[0],
              fish_atm[1]+V*di[1]]

    return newpos

def moveroutine_boids_MultiFish(pos_atm,last_pos,x,dtype=None):

    if dtype is None:
        R,O,A,V,alpha = x
    else:
        R,O,A,V,alpha = x.cpu().detach().numpy().squeeze()
    pos_last = np.nan_to_num(last_pos)
    pos_atm  = np.nan_to_num(pos_atm)
    distances = np.zeros((len(pos_atm),len(pos_atm)))
    directions = np.zeros((len(pos_atm),*pos_atm.shape))
    for i in range(len(pos_atm)):
        dirc=np.repeat([pos_atm[i]],len(pos_atm),axis=0)-pos_atm
        directions[:,i,:] = dirc
        distances[:,i]=np.sqrt(np.sum(dirc**2,axis=1))
        distances[i,i] = np.inf

    RMask = (distances<R)
    OMask = (distances<O)
    AMask = (distances<A)
    directionsR = directions.copy()
    Orientation = pos_atm-last_pos
    Orientation = np.repeat([Orientation],len(pos_atm),axis=0)
    centerOfMass = com(pos_atm)

    Attraction = np.repeat([centerOfMass-pos_atm],len(pos_atm),axis=0)

    directionsR[~RMask] = 0
    Orientation[~OMask] = 0
    Attraction[~AMask] = 0

    directionsR = np.sum(directionsR,axis=1)
    directionsR = directionsR/np.expand_dims(np.sqrt(np.sum(directionsR**2,axis=-1)),-1)
    directionsR = np.nan_to_num(directionsR) 

    
    directionsO = Orientation
    directionsO = np.sum(directionsO,axis=1)
    directionsO = directionsO/np.expand_dims(np.sqrt(np.sum(directionsO**2,axis=-1)),-1)
    directionsO = np.nan_to_num(directionsO)

    
    
    directionsA = Attraction
    directionsA = np.sum(directionsA,axis=1)
    directionsA = directionsA/np.expand_dims(np.sqrt(np.sum(directionsA**2,axis=-1)),-1)
    directionsA = np.nan_to_num(directionsA)

    
    # if R contains fish, O & A not relevant
    RMask = (directionsR != 0)
    directionsA[RMask] = 0
    directionsO[RMask] = 0
    if dtype is None:
        return pos_atm + V*(-directionsR + alpha*directionsO+(1-alpha)*directionsA)
    pos_atm = torch.tensor(pos_atm).type(dtype)
    directionsR = torch.tensor(directionsR).type(dtype)
    directionsO = torch.tensor(directionsO).type(dtype)
    directionsA = torch.tensor(directionsA).type(dtype)
    return pos_atm + x.squeeze(0)[-2]*(-directionsR + x.squeeze(0)[-1]*directionsO+(1-x.squeeze(0)[-1])*directionsA)


class BoidsMetrisch(object):
    """docstring for BoidsMetrisch"""
    def __init__(self, grid=None):
        super(BoidsMetrisch, self).__init__()
        self.grid = grid

    def x(self):
        if self.grid is None:
            return np.abs(np.random.uniform(size=(1,4)))
        x0 = np.array([np.random.uniform(l,h) for l,h in self.grid])
        return  

    def __str__(self):
        return "BoidsMetrisch"

    def ParameterNames(self):
        return ["Repulsion Zone","Orientation Zone","Atraction Zone","Theta"]

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
        Theta = parameter[:,3]
        Alpha = parameter[:,4]

        Op_pred,Or_pred= DynamicalStates(pred_data)
        Op_true,Or_true = DynamicalStates(true_data)
        sigma = 1
        fig, axs = plt.subplots(parameter.shape[-1]+4, 1,figsize=(20,50))

        pixelToCM = 0.16
        x = np.arange(len(Op_true))
        axs[0].plot(np.arange(len(Op_pred))/32,gaussian_filter1d(Op_pred,sigma),label="Op pred",color="red",linewidth=3)
        axs[1].plot(np.arange(len(Or_pred))/32,gaussian_filter1d(Or_pred,sigma),label="Or pred",color="blue",linewidth=3)
        axs[1].plot(x/32,gaussian_filter1d(Or_true,sigma),label="Or True",color="blue",alpha=0.4,linewidth=4)
        axs[0].plot(x/32,gaussian_filter1d(Op_true,sigma),label="Op True",color="red",alpha=0.4,linewidth=4)
        axs[0].set_ylabel("Rotation/Polarization",fontsize=18)
        axs[0].tick_params(axis='both', which='major', labelsize=18)
        axs[1].set_ylabel("Rotation/Polarization",fontsize=18)
        axs[1].tick_params(axis='both', which='major', labelsize=18)
        label = self.ParameterNames()
        for i in range(1,parameter.shape[-1]-1):
            if "true_params" in data:

                true_params = data["true_params"]

                try:
                    axs[i+1].plot(np.arange(len(true_params[:,i-1]))/32,
                        gaussian_filter1d(true_params[:,i-1].astype(np.float)*pixelToCM,sigma),
                        label = label[i-1]+" True",alpha=0.9,linewidth=3,color="orange")
                    axs[i+1].tick_params(axis='both', which='major', labelsize=18)
                except:
                    pass
            axs[i+1].plot(np.arange(len(parameter[:,i-1]))/32,
                gaussian_filter1d(parameter[:,i-1].astype(np.float)*pixelToCM,sigma),
                label = label[i-1]+" Predicted",linewidth=3,color="blue")
            axs[i+1].set_ylabel("Distanz in $cm$",fontsize=20)
            axs[i+1].tick_params(axis='both', which='major', labelsize=18)

        axs[-3].plot(np.arange(len(Theta))/32,gaussian_filter1d(Theta.astype(np.float),sigma),label="Theta predicted",linewidth=3,color="blue")
        axs[-2].plot(np.arange(len(Alpha))/32,gaussian_filter1d(Alpha.astype(np.float),sigma),label="Alpha predicted",linewidth=3,color="blue")
        if "true_params" in data:
            try:
                true_params = data["true_params"]
                Theta = true_params[:,3]
                Alpha = true_params[:,4]
                axs[-3].plot(np.arange(len(Theta))/32,gaussian_filter1d(Theta.astype(np.float),sigma),alpha=0.9,label="Theta True",linewidth=3,color="orange")
                axs[-2].plot(np.arange(len(Alpha))/32,gaussian_filter1d(Alpha.astype(np.float),sigma),alpha=0.9,label="Alpha True",linewidth=3,color="orange")
                axs[-3].set_ylabel("Grad",fontsize=20)
                axs[-2].set_ylabel("Grad",fontsize=20)
                axs[-2].tick_params(axis='both', which='major', labelsize=18)
                axs[-1].tick_params(axis='both', which='major', labelsize=18)
                axs[-3].tick_params(axis='both', which='major', labelsize=18)
            except:
                pass
        axs[-1].plot(np.arange(len(data["best_error"])),data["best_error"],label="l2 error",linewidth=3)
        for i in range(len(axs)):
            axs[i].legend(loc="upper right",prop={'size': 18})
            axs[i].set_xlabel("Zeit in $S$",fontsize=20)
            axs[i].grid(True)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
        axs[-1].set_xlabel("Sequenz",fontsize=20)
        axs[-1].set_ylabel("l2-error",fontsize=20)

        path = path2file.replace(".npz","")
        if not os.path.exists(path):
            os.makedirs(path)


        plt.savefig(os.path.join(path,"parameter.png"))
        #plt.show()

    def __call__(self,pos_atm,last_pos,x):
        pos_last = torch.tensor(last_pos)
        pos_atm = torch.tensor(pos_atm)
        pos_atm = torch.nan_to_num(pos_atm)
        pos_last = torch.nan_to_num(pos_last)

        d = pos_atm.repeat(len(pos_atm),1).reshape(len(pos_atm),len(pos_atm),2)
        # transponiert und subtraktion = position von fischen - position von fisch i
        # also richtungsvektor von fisch i nach Fischen
        # Skalierung mit R = alle innerhalb R einheitskreis
        dir_scaled = (d-d.transpose(1,0))/x[0]
        # distanzen berechnen
        dis_scaled = torch.sqrt(torch.sum(dir_scaled**2,axis=-1))


        # Diagonale muss auf True gesetzt werden, da nicht teil der nachbarschaft
        diag = torch.eye(len(pos_atm))

        # Nur Fische innerhalb R!
        idx20R = dis_scaled>1
        idx20R = (idx20R+diag)==1

        # Vektoren Skalieren 
        norm_dir   = torch.nan_to_num(dir_scaled/dis_scaled.unsqueeze(-1))
        norm_dir_blackspot = norm_dir.detach().numpy().copy()

        #Orientierung berechnen

        orient = (pos_atm-pos_last).repeat(len(pos_atm),1).reshape(len(pos_atm),len(pos_atm),2).transpose(1,0)
        orient = torch.nan_to_num(orient/torch.sqrt(torch.sum(orient**2,axis=-1)).unsqueeze(-1))

        #blindspot berechnen
        n = norm_dir_blackspot
        no = orient.detach().numpy()
        angles = np.zeros((len(pos_atm),len(pos_atm)))

        for i in range(len(n)):
            angles[:,i] = np.rad2deg(np.arccos((np.dot(np.abs(n[i]),np.abs(no[i]).transpose(1,0))).diagonal()))

        wod = (x[4]//2)
        
        blindspot = (angles > wod)
        idx20R += blindspot

        # Distanzen der Fische die nicht in R sind auf 0 setzen
        norm_dir[idx20R] = 0
        #norm_dirR = norm_dir
        # Mittlere Richtung bestimmen
        #len_dir = torch.sum(~idx20,axis=0)-1
        sum_dir = torch.sum(norm_dir,axis=0)

        d_ir = torch.nan_to_num(sum_dir)





        dir_scaled = (d-d.transpose(1,0))/x[1]

        dis_scaled = torch.sqrt(torch.sum(dir_scaled**2,axis=-1))

        idx20O = dis_scaled>1
        idx20O += blindspot
        idx20O = (idx20O+diag)==1
        #Überlappungen Rausnehmen
        idx20O[(idx20R==idx20O)] = True


        # Orientierung darf die eigene Orientierung beinhalten
        orient[idx20O] = 0



        sum_or = torch.sum(orient,axis=1)

        d_io = torch.nan_to_num(sum_or)


        # Attraction berechnen

        d = pos_atm.repeat(len(pos_atm),1).reshape(len(pos_atm),len(pos_atm),2)

        # transponiert und subtraktion = position von fischen - position von fisch i
        # also richtungsvektor von fisch i nach fischen
        # Skalierung mit R = alle innerhalb R einheitskreis
        dir_scaled = (d-d.transpose(1,0))/x[2]

        # distanzen berechnen
        dis_scaled = torch.sqrt(torch.sum(dir_scaled**2,axis=-1))

        # Nur Fische innerhalb R!
        idx20A = dis_scaled>1 
        idx20A+= blindspot
        idx20A = (idx20A+diag)==1
        #Überlappungen Rausnehmen
        idx20A[idx20O==idx20A] = True
        idx20A[idx20R==idx20A] = True

        # Vektoren Skalieren 
        norm_dir   = torch.nan_to_num(dir_scaled/dis_scaled.unsqueeze(-1))

        # Distanzen der Fische die nicht in A sind auf 0 setzen
        norm_dir[idx20A] = 0


        d_ia = -torch.nan_to_num(torch.sum(norm_dir,axis=0))

        set_toZero_ia = torch.sum(d_ia**2,axis=1) > 0
        set_toZero_io = torch.sum(d_io**2,axis=1) > 0

        both = (set_toZero_ia) & (set_toZero_io)
        scale = torch.ones_like(d_io)
        scale[both]=0.5

        """
        d_ir = d_ir.detach().numpy()
        d_io = d_io.detach().numpy()
        d_ia = d_ia.detach().numpy()
        scale = scale.detach().numpy()
        d_ir = d_ir/np.expand_dims(np.sqrt(np.sum(d_ir**2,axis=-1)),-1)
        d_io = d_io/np.expand_dims(np.sqrt(np.sum(d_io**2,axis=-1)),-1)
        d_ia = d_ia/np.expand_dims(np.sqrt(np.sum(d_ia**2,axis=-1)),-1)
        """
        #desired direction
        di = d_ir + scale*d_io + scale*d_ia 
        
        #di = di/((torch.sqrt(torch.sum(di**2,-1))).unsqueeze(-1))

        di = di.detach().numpy()
        ci = di.copy()
        
        last_direction = pos_atm-pos_last
        #last_direction = last_direction/torch.sqrt(torch.sum(last_direction**2,axis=-1)).unsqueeze(-1)
        for i in range(len(di)):
            vi = turnmax(last_direction[i],di[i],x[3])
            if np.isnan(np.rad2deg(angle_between(last_direction[i],vi))):
                vi = last_direction[i]
            di[i] = vi
        if np.isnan(di).any():
            print(di)
            print(ci)

        return (pos_atm+di).detach().numpy()

