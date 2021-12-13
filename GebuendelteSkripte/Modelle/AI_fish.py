import numpy as np
import cv2
import math
from Utils.Common import *

class AIFish:
    def __init__(self,pos,rad,move_routine=None,angle = None,direction = None):
        self.pos = pos
        self.rad = rad
        self.color = (255,255,255)
        self.velcity_sigma = 0.5
        self.dead = False
        self.history = []
        self.snapshot()
        self.move_routine = move_routine
        if direction is None:
            self.direction = [0,0]
            while float(VecLen(self.direction)) == 0.0:
                self.direction = normalize_Vec(np.random.uniform(-1,1,2))
                

        else:
            self.direction = direction

        self.angle = angle
        self.last_pos = pos

    def insideBoundaries(self,h,w):
        if self.pos[0] >= 0 and h >self.pos[0] \
        and self.pos[1] >= 0 and w >self.pos[1]:
            return True
        return False

        
    def snapshot(self):
        self.history.append([self.pos[0],self.pos[1]])
    

    def move(self,direction,velocity,mu=0,sigma=1):
        
        direction = self.direction
        if self.move_routine is None:
            x = np.abs(np.random.normal(mu, sigma, 1))
            y = np.random.normal(mu, sigma, 1)
            new_pos = [self.pos[0],self.pos[1]]
            velocity = velocity + np.random.normal(0, self.velcity_sigma, 1)
            
            new_pos[0] += int((direction[0] + x)*velocity)
            new_pos[1] += int((direction[1] + y)*velocity)
            self.pos = tuple(new_pos)

        else:
            self.pos = self.move_routine(self.pos)

        self.snapshot()
    
    def move2Pos(self,newpos):
        self.pos = (int(newpos[0]),int(newpos[1]))
        self.snapshot()

    def getLastPos(self):
        if len(self.history) < 2:
            return None

        return self.history[-2]

    def getPos(self):
        return self.pos
    """
    def fish2img(self,img,color=(0,255,0)):
        
        x,y = img.shape[:2]
        if self.pos[0]-self.rad > y or self.pos[1]-self.rad>x:
            self.dead = True
            
        img = cv2.circle(img, self.pos, self.rad, self.color, -1)
        img = cv2.circle(img, self.pos, self.rad, color, 1)
        img[self.pos[1]-1:self.pos[1]+2,self.pos[0]-1:self.pos[0]+2] = color
        
        return img
    """
    def fish2img(self,img,dontFill=False,color=(255,255,255),thickness=1):
        
        def drawTriangle(img,rad):
            halfrad = rad//2
            triangle = np.array([[0-halfrad,0-halfrad],[0-halfrad,0+halfrad],[0+rad,0]])

            def dotproduct(v1, v2):
                return sum((a*b) for a, b in zip(v1, v2))

            def length(v):
                return math.sqrt(dotproduct(v, v))

            def angle(v1, v2):
                return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


            direction      = np.array([self.pos[0] - self.last_pos[0],self.pos[1] - self.last_pos[1]])
            if np.sum(direction) == 0:
                direction = np.random.uniform(10,size=2) - 5

            direction = direction / np.linalg.norm(direction)*rad
            angle2Rot = angle(direction,[1,0])
            if dotproduct(direction,[0,1]) < 0:
                angle2Rot *=-1

            rotatet_Tri = []
            for pts in triangle:
                rotatet_Tri.append(rotate([0,0],pts,angle2Rot) + self.pos)
            triangle = rotatet_Tri
            triangle = np.int32([triangle])
            
            return cv2.polylines(img, triangle, isClosed=True, color=color, thickness=thickness)

            

        fill = -1
        rad = self.rad
        
        if color is None:
            color = self.color
            
        if dontFill:
            fill = 1
            rad  = 5
        x,y = img.shape[:2]
        if self.insideBoundaries(y,x):
            img = drawTriangle(img,rad)
        return img

    def lastPos2Img(self,img,color=(255,255,255),thickness=1,length=None):
        if length is None:
            length = len(self.history)
        for i in range(len(self.history)-1,len(self.history)-length+1,-1):
            if i<1:
                break

            x1,y1 = self.history[i]
            x2,y2 = self.history[i-1]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)

        return img

    def alive(self):
        return not self.dead

    