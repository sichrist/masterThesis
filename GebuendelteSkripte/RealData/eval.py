import sys
sys.path.append("../")
from Modelle.PositiveWeightedDistances import Positive_weighted_distances
from Modelle.Boids import BoidsTopologisch,BoidsMetrisch
from SimulateFakeData import VisualizeFakeData
import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Common import *

def load_data(path):
    return np.load(path,allow_pickle=True)

def visIt(data,path2file):
    parameter = data["best_params"]
    true_data = data["true_pos"]
    if "Boids" in path2file:
        function = BoidsMetrisch()
    else:
        function = Positive_weighted_distances()

    path = path2file.replace(".npz","")
    if not os.path.exists(path):
        os.makedirs(path)

    vis = VisualizeFakeData(true_data,
        function,
        parameter,
        dsize=(4000,4000),
        filename=os.path.join(path,"Video"),split=True)

    while vis():
        continue
    return vis.predicted_pos
def plotIt(data,path2file,pred_pos = None):

    if "Boids" in path2file:
        BoidsMetrisch().plotIt(data,path2file,pred_pos=pred_pos)
    else:
        Positive_weighted_distances().plotIt(data,path2file)

def main():
    DEFAULT_PATH="PSO_Approximation"
    #DEFAULT_PATH="RMD_Approximation"
    filename = "zebrafish_trajectories_60_1_BoidsMetrisch_l2error__RS_False11-15-2021-13-35-06.npz"
    """

    path = os.path.join(DEFAULT_PATH,filename)
    data = load_data(path)
    visIt(data,filename)
    #plotIt(data,filename)
    """
    i = 0
    pwds = []
    for file in os.listdir(DEFAULT_PATH):
        if file.endswith(".npz"):
            print(i,")",os.path.join(DEFAULT_PATH, file))
            i+=1
            pwds.append(os.path.join(DEFAULT_PATH, file))


    for i in range(len(pwds)):
        path = pwds[int(i)]
        data = load_data(path)
        filename = path.split("/")[-1]
        pred_pos = visIt(data,filename)
        plotIt(data,filename,pred_pos=pred_pos)





if __name__ == '__main__':
    main()