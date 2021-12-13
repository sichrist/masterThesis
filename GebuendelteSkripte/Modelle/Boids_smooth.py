from Utils.Common import *

def moveroutine_boids_func(pos_atm,last_pos,alpha,R,O,A,V):
    newpos = []
    
    for i,p1 in enumerate(pos_atm):
        
        di = []
        for j,p2 in enumerate(pos_atm):
            if i == j:
                continue
            d = dist(p1,p2)
        
            p1p2 = normalize_Vec(direction(p2,p1))
            orientation = normalize_Vec(direction(pos_atm[j],last_pos[j]))
                                 
            di.append(FR(d,R)*(-p1p2) + alpha * (orientation *FO(d,R,O,A))  + (1-alpha)*(p1p2 * FA(d,O,A)))

        di = np.sum(di,axis=0)

        di = normalize_Vec(di)
        if np.isnan(di[0]) or np.isnan(di[1]):
            di = [0,0]
        newpos.append([p1[0]+V*di[0],
                       p1[1]+V*di[1]])
        

        pos_atm[i] = newpos[-1]
    return np.array(newpos)



def moveroutine_boids_SingleFish_func(pos_atm,last_pos,R,O,A,V,alpha,ID):
    newpos = []
    
    fish_atm = pos_atm[ID]
    fish_last = last_pos[ID]
    last_pos = np.nan_to_num(last_pos)
    pos_atm  = np.nan_to_num(pos_atm)
    di = []
    for j,p2 in enumerate(pos_atm):
        if ID == j:
            continue
        d = dist(fish_atm,p2)
    
        p1p2 = normalize_Vec(direction(p2,fish_atm))
        orientation = normalize_Vec(direction(pos_atm[j],last_pos[j]))
                             
        di.append(FR(d,R)*(-p1p2) + alpha * (orientation *FO(d,R,O,A))  + (1-alpha)*(p1p2 * FA(d,O,A)))

    di = np.sum(di,axis=0)

    di = normalize_Vec(di)
    if np.isnan(di[0]) or np.isnan(di[1]):
        di = [0,0]
    newpos.append([fish_atm[0]+V*di[0],
                   fish_atm[1]+V*di[1]])
    

    return np.array(newpos)
