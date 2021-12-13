import cv2
import numpy as np
from time import sleep

class Linker(object):
    """docstring for Linker"""
    def  __init__(self, Video,blobs):
        super(Linker, self).__init__()
        self.Video = Video
        self.blobs = blobs


    def calculateProbabilities(self,individuals,objects):
        print(individuals)

        for o in objects:
            print(o.index(),end=",")
        print()
        pass

    def start(self,expected):
        bloblist = self.blobs[0]

        self.individuals = [[]]
        for i in range(expected):

            try:
                self.individuals[0].append((i,bloblist[i].index()))
            except:
                self.individuals[0].append((i,None))
                
        for objects in self.blobs[1:]:
            self.calculateProbabilities(self.individuals,objects)
                



    