from videoOPS import *
import numpy as np



IDS = [i for i in range(300000)]
import copy


def cluster_fish(img):

    if len(img.shape) > 2:
        img = to_gray(img)

    img = to_binary(img,threshold = 120)

    return img


IDS = [i for i in range(3000000)]

class Fish:
    def __init__(self,pos,texture=None,fish=None,id =None):
        self.pos = pos
        if id is None:
            self.id  = IDS.pop()
        else:
            self.id = id
        self.color = (255,255,255)
        self.taken = False
        self.texture = texture
        self.fish = fish
        


    def getPos(self):
        return self.pos


    def getID(self):
        return self.id

    
    def insideBoundaries(self,h,w):
        if self.pos[0] >= 0 and h >self.pos[0] \
        and self.pos[1] >= 0 and w >self.pos[1]:
            return True
        return False


    def fish2img(self,img,dontFill=False,color=None):
        
        fill = -1
        rad = self.rad
        
        if color is None:
            color = self.color
            
        if dontFill:
            fill = 1
            rad  = 5
        x,y = img.shape[:2]
        if self.insideBoundaries(y,x):            
            img = cv2.circle(img, self.pos, rad, color, -1)
            img = cv2.circle(img, self.pos, rad, (0,255,0), 1)
            img[self.pos[1]-1:self.pos[1]+2,self.pos[0]-1:self.pos[0]+2,:] = [0,255,0]
        return img


    def isTaken(self):
        return self.taken
    
    def index(self):
        return self.pos[0],self.pos[1]


    def move(self,direction,velocity,mu=0,sigma=1):
        
        if self.move_routine is None:

            x = np.abs(np.random.normal(mu, sigma, 1))
            y = np.random.normal(mu, sigma, 1)
            new_pos = [self.pos[0],self.pos[1]]
            velocity = velocity + np.random.normal(0, self.velcity_sigma, 1)
            
            new_pos[0] += int((direction[0] + x)*velocity)
            new_pos[1] += int((direction[1] + y)*velocity)
            self.pos = tuple(new_pos)
        else:

            self.pos = self.move_routine(self.pos,self.angle)

    def moveViaFlow(self,flow,window=10):
        x,y    = self.pos
        
        if self.fish is not None:
            part   = flow[self.fish]
            mean   = np.nan_to_num(part)
            mean   = np.mean(part,axis=0)

        else:
            part   = flow[x-window:x+window+1,y-window:y+window+1]
            mean   = np.mean(part,axis=(0,1))
            mean   = np.nan_to_num(mean)

        self.pos = (int(mean[0] + self.pos[0]),int(mean[1] + self.pos[1]))

        
    def dist2fish(self,fish):
        x,y = self.pos
        p_x,p_y = fish.pos
        return np.sqrt((x - p_x)**2 + (y - p_y)**2)