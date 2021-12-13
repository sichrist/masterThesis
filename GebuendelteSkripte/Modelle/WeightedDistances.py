from Utils.Common import *
import numpy as np

def weighted_distances(pos_atm,pos_last,omega1,omega2,omega3):
    pos_last = np.nan_to_num(pos_last)
    pos_atm  = np.nan_to_num(pos_atm)
    def getNeareast(pos,id):
        f = pos[id]
        distances = [(dist(pos[i],f),i) for i in range(len(pos)) if i != id]
        distances = sorted(distances, key=lambda tup : tup[0])
        return distances[0]
    
    v1 = []
    v2 = []
    v3 = []
    center_of_school = com(pos_atm)
    v2 = np.array([normalize_Vec(np.sum(pos_atm - pos_last,axis=(0))/len(pos_atm))]*len(pos_last),dtype=np.float32)

    for i in range(len(pos_atm)):
        d,id   = getNeareast(pos_atm,i)
        dirc   = direction(pos_atm[i],pos_atm[id])
        v1.append(normalize_Vec(dirc))
        v3.append(normalize_Vec(direction(center_of_school,pos_atm[id])))
    v1 = np.array(v1,dtype=np.float32)
    v3 = np.array(v3,dtype=np.float32)
    vec = omega1*v1 + omega2*v2 + omega3*v3
    return pos_atm+vec


def weighted_distances_SingleFish(pos_atm,pos_last,omega1,omega2,omega3,ID):
    pos_last = np.nan_to_num(pos_last)
    pos_atm  = np.nan_to_num(pos_atm)

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

    v1.append(normalize_Vec(dirc))
    v3.append(normalize_Vec(direction(center_of_school,pos_atm[id])))


    v1 = np.array(v1,dtype=np.float32)
    v3 = np.array(v3,dtype=np.float32)

    vec = omega1*v1 + omega2*v2 + omega3*v3

    return pos_atm[ID]+vec