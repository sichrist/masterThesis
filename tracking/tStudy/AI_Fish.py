import numpy as np
import cv2 
import os
import math
IDS = [i for i in range(3000,-1,-1)]

import copy

def VecLen(V):
    return np.sqrt(V[0]**2 + V[1]**2) 

def dist(p1,p2):
    return VecLen(np.array(p1)-np.array(p2))

def normalize_Vec(V):
    if np.sum(VecLen(V)) == 0:
        return np.array([0,0])
    return V / VecLen(V)

def rotate(origin, point, angle):
    import math
    #angle = math.degrees(angle)
    #print(angle)

    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([np.int(qx), np.int(qy)])

class AIFish:
    def __init__(self,pos,rad,move_routine=None,angle = None):
        self.pos = pos
        self.id  = IDS.pop()
        self.rad = rad
        self.color = (255,255,255)
        self.velcity_sigma = 0.5
        self.taken = False
        self.move_routine = move_routine
        self.angle = angle
        self.last_pos = pos


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
        
        def drawTriangle(img,rad):
            halfrad = rad//5
            triangle = np.array([[0-halfrad,0-halfrad],[0-halfrad,0+halfrad],[0+rad,0]])


            triangle = np.array([[-5,0],[-4,0.1],[-3,0.2],[-2,0.4],[-1,1],[0,1.0],[1,0.3],[2,0.1]
                                ,[-5,0],[-4,-0.1],[-3,-0.2],[-2,-0.4],[-1,-1],[0,-1.0],[1,-0.3],[2,-0.1]])*rad

            def dotproduct(v1, v2):
                return sum((a*b) for a, b in zip(v1, v2))

            def length(v):
                return math.sqrt(dotproduct(v, v))

            def angle(v1, v2):
                return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

            


            if self.last_pos[0] != self.pos[0] or self.last_pos[1] != self.pos[1]:

                direction      = np.array([self.pos[0] - self.last_pos[0],self.pos[1] - self.last_pos[1]])
                direction = direction / np.linalg.norm(direction)*rad
                angle2Rot = angle(direction,[1,0])
                if dotproduct(direction,[0,1]) < 0:
                    angle2Rot *=-1

                rotatet_Tri = []
                for pts in triangle:
                    rotatet_Tri.append(rotate([0,0],pts,angle2Rot) + self.pos)
                triangle = rotatet_Tri
                triangle = np.int32([triangle])
                return cv2.polylines(img, triangle, isClosed=True, color=(255, 255, 255), thickness=4)

                """
                direction = np.int32(direction)
                img = cv2.arrowedLine(img, 
                    tuple(np.array(self.pos)-direction), 
                    tuple(np.array(self.pos)+direction),
                    (255,0,0),
                    2,
                    tipLength = 0.5) 
                """


            

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
            #img = cv2.circle(img, self.pos, rad, color, -1)
            #img = cv2.circle(img, self.pos, rad, (0,255,0), 1)
            #img[self.pos[1]-1:self.pos[1]+2,self.pos[0]-1:self.pos[0]+2,:] = [0,255,0]
        return img


    def isTaken(self):
        return self.taken


    def index(self):
        return self.pos[0],self.pos[1]


    def move(self,direction,velocity,mu=0,sigma=1):

        self.last_pos = self.pos
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


    def moveViaFlow(self,flow,window=None):
        if window == None:
            window = self.rad
        x,y    = self.pos
        part   = flow[y-window:y+window+1,x-window:x+window+1]
        mean   = np.mean(part,axis=(0,1))
        mean   = np.nan_to_num(mean)
        self.pos = (int(mean[0] + self.pos[0]),int(mean[1] + self.pos[1]))
        
        
    def dist2fish(self,fish):
        x,y = self.pos
        p_x,p_y = fish.pos
        return np.sqrt((x - p_x)**2 + (y - p_y)**2)


class SchoolOfFishStream_AI:
    def __init__(self,img_shape  = (480,640,3),
                      radius     = (15,20),
                      start_area = (20,640),
                      velocity   = 0.5,
                      direction  = [1,0],
                      nbr        = 10,
                      savename   = None,
                      seed       = None,
                      move_rout  = None,
                      angle      = None,
                      init_func  = None,
                      frames     = None):


        self.img_shape  = img_shape
        self.radius     = radius
        self.start_area = start_area
        self.velocity   = velocity
        self.direction  = direction
        self.seed       = seed
        self.plain_img  = None
        self.school     = []
        self.nbr        = nbr
        self.frames     = frames
        self.angle      = angle
        self.frame      = 0
        np.random.seed(self.seed)   
        self.move_routine = move_rout
        self.init_fish(init_function=init_func)
        self.savename = savename
        self.basedir = "SchoolOfFishStream_AI"
        # if at least one fish inside boundaries
        self.inImage   = False
        self.history   = {}
        self.images_out= 0

        if self.savename is not None:

            if not os.path.exists(self.basedir):
                os.mkdir(self.basedir)

            if not os.path.exists(os.path.join(self.basedir,self.savename)):
                os.mkdir(os.path.join(self.basedir,self.savename))
            self.img_folder = os.path.join(self.basedir,self.savename)
            self.video = cv2.VideoWriter(os.path.join(self.img_folder,"Video.avi") ,cv2.VideoWriter_fourcc(*'DIVX'), 30, (self.img_shape[:2]))

        
    def init_fish(self,init_function=None):
        x,y       = self.start_area
        rmin,rmax = self.radius
        if init_function is None:
            pos_y = np.random.uniform(low=self.radius[1], high=y-self.radius[0], size=self.nbr).astype(np.int)
            pos_x = np.random.uniform(low=self.radius[1], high=x-self.radius[0], size=self.nbr).astype(np.int)
        else:
            pos_y,pos_x = init_function()

        r = np.random.uniform(low=rmin, high=rmax, size=self.nbr).astype(np.int)
        self.school = [AIFish(pos,r[i],move_routine=self.move_routine,angle=self.angle) for i,pos in enumerate(zip(pos_x,pos_y))]


    def getMeanImg(self):
        return np.zeros(self.img_shape)


    def snapshot(self):
        for f in self.school:
            id_ = f.getID()

            if id_ not in self.history:
                self.history[id_] = []
                
            c = copy.copy(f)
            self.history[id_].append((c,self.frame))
                
            
    def getHistory(self):
        return self.history


    def fish_inside_view(self):
        sum_ = 0
        
        for f in self.school:
            isin = f.insideBoundaries(self.img_shape[1],self.img_shape[0])
            sum_ += isin
        
        if sum_ > 0:
            return True
        return False


    def update(self):

        if self.plain_img is None:
            self.plain_img = np.zeros(self.img_shape)
            img = self.plain_img.copy()
            
            for f in self.school:
                f.fish2img(img)
            return img
        
        img = self.plain_img.copy()
        for f in self.school:
            f.move(self.direction,self.velocity)
            f.fish2img(img)

        return img

    def saveTrajectories(self):
        trajectories = np.zeros((self.frames,len(self.history),2),dtype=np.uint)
        keys = self.history.keys()

        for key in self.history:
            for f,frame in self.history[key]:
                trajectories[frame-1,key] = list(f.index()) 
        np.save(os.path.join(self.img_folder,"Video"),trajectories)

        
    def __call__(self):

        if self.frames is not None:
            if self.frame >= self.frames:
                self.frame = 0
                if self.savename is not None:
                    self.saveTrajectories()
                return None

            self.frame += 1

        if not self.fish_inside_view():
            return None
                    
        img = self.update()

        self.video.write(img.astype(np.uint8))
        self.snapshot()
        self.images_out += 1
        return img


    def __del__(self):
        if self.video is not None:
            cv2.destroyAllWindows()
            self.video.release()


def max_area2move(flow):
    """
        calculate the max distance a fish moves (according to its flow)
    """
    magnitude = np.sqrt((np.sum(flow**2,axis=-1))).flatten()
    idx = np.where(magnitude > 0)
    return magnitude[idx].max()

def mean_area2move(flow):
    """
        calculate average distance a fish moves
        add 2*std to this distance
    """
    magnitude = np.sqrt((np.sum(flow**2,axis=-1))).flatten()
    idx = np.where(magnitude > 0)
    return magnitude[idx].mean()+magnitude[idx].std()