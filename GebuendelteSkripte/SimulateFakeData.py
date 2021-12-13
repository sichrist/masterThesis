import numpy as np
from Modelle.WeightedDistances import *
from Modelle.PositiveWeightedDistances import *
from Modelle.Boids import *
from Modelle.Boids_smooth import *
import cv2
from time import sleep
np.random.seed(None)
import os
from Utils.environment import in_notebook
from Utils.Common import show_in_Notebook
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/schristo/.anaconda3/envs/py39/lib/python3.9/site-packages/cv2/qt/plugins/platforms"
def path2img(pos,img,color=(255,255,255),maxlen=10):

    if len(pos)<2:
        return img
    
    for i in range(len(pos)-1,len(pos)-maxlen,-1):
        if i <0:
            break
        for a,b in zip(pos[i],pos[i-1]):
            a = np.nan_to_num(a)
            b = np.nan_to_num(b)
            x = (int(a[0])+500,int(a[1])+500)
            y = (int(b[0])+500,int(b[1])+500)
            if np.sum(x) == 0 or np.sum(y)==0:
                continue
            img = cv2.line(img,x,y,color,2) 
    return img

def dots2frame(img,tra,size=10,color = (255,255,255)):

    for fid,a in enumerate(tra):
        a = np.nan_to_num(a)
        x,y = int(a[0])+500,int(a[1])+500
        cv2.circle(img,(x,y),size,color,-1)
    return img

def randomPositionsOnCircle(nbr,size,RadiusRange,std=100):
    x0 = np.array([size[0]//2,size[1]//2])

    pos = np.random.uniform((size[0]//2)-std,(size[1]//2)+std,size=(nbr,2))
    rdPosCircle = []
    for p in pos:
        direction = (x0 - p)/np.expand_dims(np.sqrt(np.sum((x0 - p)**2,axis=-1)),-1)
        Radius = np.random.uniform(RadiusRange[0],RadiusRange[1],1)
        p = x0+(direction*Radius)
        x,y = int(p[0]),int(p[1])
        rdPosCircle.append([x,y])
    return np.array(rdPosCircle)

class GenerateFakeData(object):
    """docstring for GenerateFakeData"""
    def __init__(self,
                    function,
                    randomRange=[(0,5),(0.5,1.0),(0.8,1.0),(0.9,1.0)],
                    randomParameter=True,
                    initialPositions=None,
                    nbr_fish=10,
                    dsize=(2000,2000),
                    std=50,
                    show=True,
                    filename = None):
        self.windowname = "FakeData"
        super(GenerateFakeData, self).__init__()
        self.randomParameter = randomParameter
        self.paramlist       = []
        self.function        = function
        self.randomRange     = randomRange
        self.positions       = []
        self.nbr_fish        = nbr_fish
        self.show            = show
        self.dsize           = dsize
        self.initialPositions= initialPositions
        self.filename = filename
        if self.initialPositions is None:
            #self.pos = np.array([dsize[0]//2,dsize[1]//2]*self.nbr_fish).reshape(self.nbr_fish,2)
            #self.pos = randomPositionsOnCircle(self.nbr_fish,dsize,(300,400))
            #self.positions.append(self.pos)
            #self.positions.append(self.pos+[2,0])
            self.pos = np.random.uniform(dsize[0]//2-std,dsize[1]//2+std,size=(self.nbr_fish,2))
            self.positions.append(self.pos)
            self.positions.append(self.pos+[0.1,0.1])
            #self.positions.append(randomPositionsOnCircle(self.nbr_fish,dsize,(300,400)))
            #self.positions.append(self.pos+np.random.uniform(-100,100,size=(self.nbr_fish,2)))
            #self.positions.append(self.pos+np.random.uniform(-1000,1000,size=(self.nbr_fish,2)))
        else:
            if len(self.initialPositions[0]) != len(self.initialPositions[1]):
                print("initialPositions !=")
                exit(-1)
            self.nbr_fish = len(self.initialPositions[0])
            self.pos = self.initialPositions[-1]
            self.positions.append(self.initialPositions[-2])
            self.positions.append(self.initialPositions[-1])
        self.params = []
        if self.show == True and in_notebook()== False:
            cv2.namedWindow(self.windowname)        # Create a named window
            cv2.moveWindow(self.windowname, 40,30)
        self.i = 0

    def reset(self):
        self.params = []
        self.positions.append(pos)
        self.positions.append(pos+[5,5])


    def getParams(self):
        return self.params


    def dots2frame(self,img,size=10,color = (0,0,0)):
        tra = np.array(self.positions[-1])

        for fid,a in enumerate(tra):
            a = np.nan_to_num(a)
            x,y = int(a[0]),int(a[1])
            cv2.circle(img,(x,y),size,color,-1)
        return img

    def __str__(self):
        return "GenerateFakeData"

    def path2img(self,img,color=(0,0,0),maxlen=10000):
        pos = self.positions
        if len(pos)<2:
            return img
        
        for i in range(len(pos)-1,len(pos)-maxlen,-1):
            if i <1:
                break
            for a,b in zip(pos[i],pos[i-1]):
                a = np.nan_to_num(a)
                b = np.nan_to_num(b)
                x = (int(a[0]),int(a[1]))
                y = (int(b[0]),int(b[1]))
                if np.sum(x) == 0 or np.sum(y)==0:
                    continue
                img = cv2.line(img,x,y,color,2) 
        return img

    def show_image(self):
        img = np.ones((*self.dsize,3),dtype=np.uint8)*150

        img = self.dots2frame(img)
        img = self.path2img(img)

        if not in_notebook():
            cv2.imshow(self.windowname,cv2.resize(img, self.dsize, interpolation = cv2.INTER_AREA))
        else:
            show_in_Notebook(img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            if self.filename is not None:
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,500)
                fontScale              = 1
                fontColor              = (0,0,0)
                thickness              = 1
                lineType               = 2

                txt = str(self.randomRange)
                cv2.putText(img,txt, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

                cv2.imwrite(self.filename+".png",img)
            exit(0)
        sleep(1/30)

    def __len__(self):
        return len(self.positions)

    def __call__(self):
        self.i+=1
        if self.randomParameter or len (self.params) == 0:
            params = []
            for (l,h )in self.randomRange:
                params.append(np.random.uniform(low=l,high=h,size=1)[0])
                #params.append(np.random.normal(h-l/2,0.1,size=1)[0])
        else: 
            params = self.params[-1]


        self.positions.append(self.function(self.positions[-1],self.positions[-2],params))

        self.params.append(params)

        if self.show:
            img = self.show_image()
        return self.positions[-1]

    def __del__(self):
        cv2.destroyAllWindows()

class VisualizeFakeData(object):
    """docstring for VisualizeFakeData"""
    def __init__(self, TrueData,function,parameterlist,dsize=(4000,4000),filename = None,split=False):
        super(VisualizeFakeData, self).__init__()
        self.function = function
        self.TrueData = TrueData
        self.parameterlist = parameterlist
        self.i = 1
        self.predicted_pos = []
        self.dsize = dsize
        self.filename = filename
        self.out = None
        self.split = split
        if self.filename is not None:
            if split:
                self.out = cv2.VideoWriter(self.filename+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 5, (2000,1000))#(dsize[0]*2,dsize[1]))
            else:
                self.out = cv2.VideoWriter(self.filename+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 5, dsize)
        self.last = self.TrueData[self.i-1]
        self.atm  = self.TrueData[self.i]
        self.TrueData = self.TrueData[2:]
        self.predicted_pos = [self.last,self.atm]
        self.trajectories = None
    def simulate(self,last,atm,param):
        pred = self.function(atm,last,param)
        self.predicted_pos.append(pred)
        self.last = self.atm
        self.atm = self.predicted_pos[-1]

    def __call__(self):
        if self.i >= len(self.parameterlist):
            print(len(self.TrueData))
            print(len(self.predicted_pos))
            print()
            return
        true = self.TrueData[self.i+1]
        

        self.simulate(self.last,self.atm,self.parameterlist[self.i-1])
        self.i+=1
        img = np.zeros((*self.dsize,3))

        if self.split == True:
            img2 = np.zeros_like(img)

        img = dots2frame(img,self.TrueData[self.i],size=3)
        img = path2img(self.TrueData[:len(self.predicted_pos)+1],img)
        if self.split == True:


            img2 = dots2frame(img2,self.predicted_pos[-1],color=(255,0,255),size=3)
            img2 = path2img(self.predicted_pos,img2,color=(0,255,255))
        else:
            img = dots2frame(img,self.predicted_pos[-1],color=(255,0,255),size=3)
            img = path2img(self.predicted_pos,img,color=(0,255,255)) 

        if self.trajectories is None:
            self.img_keep = img.copy()
            self.img2_keep = img2.copy()
            self.trajectories = []
            self.img2show = np.concatenate((self.img_keep,self.img2_keep),axis=1)
        else:

            self.img_keep = path2img(self.TrueData[:len(self.predicted_pos)+1],self.img_keep)
            #self.img_keep = dots2frame(self.img_keep,self.TrueData[self.i],size=3)
            self.img2_keep = path2img(self.predicted_pos,self.img2_keep,color=(255,255,255))
            #self.img2_keep = dots2frame(self.img2_keep,self.predicted_pos[-1],color=(255,255,255),size=3)
            self.img2show = np.concatenate((self.img_keep,self.img2_keep),axis=1)
        if self.split == True:
            img = np.concatenate((img,img2),axis=1)
        if self.out is not None:
            img2save = cv2.resize(img, (2000,1000), interpolation = cv2.INTER_AREA).astype(np.uint8)
            self.out.write(img2save)
        if self.split == True:
            pass
            #cv2.imshow('image',cv2.resize(img, (2000,1000), interpolation = cv2.INTER_AREA))
        #if self.trajectories is None:
        #    self.trajectories = 
        else:
            #cv2.imshow('image',cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA))
            cv2.imshow('image',cv2.resize(self.img2show, (1000,1000), interpolation = cv2.INTER_AREA))
            print("SHOW")
        #cv2.imshow('image',cv2.resize(self.img2show, (2000,1000), interpolation = cv2.INTER_AREA))
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            exit(0)
        #sleep(1/30)
        return True
    """
    def __call__(self):

        if self.i >= len(self.parameterlist):
            return None
        last = self.TrueData[self.i-1]
        atm  = self.TrueData[self.i]
        true = self.TrueData[self.i+1]
        self.simulate(last,atm,self.parameterlist[self.i])
        self.i+=1
        img = np.zeros((*self.dsize,3))
        if self.split == True:
            img2 = np.ones_like(img)
        img = dots2frame(img,self.TrueData[self.i+1],size=6)

        img = path2img(self.TrueData[:len(self.predicted_pos)+1],img)
        if self.split == True:
            img2 = dots2frame(img2,self.predicted_pos[-1],color=(255,0,255),size=5)
            img2 = path2img(self.predicted_pos,img2,color=(0,255,255))
        else:
            img = dots2frame(img,self.predicted_pos[-1],color=(255,0,255),size=5)
            img = path2img(self.predicted_pos,img,color=(0,255,255)) 

        if self.split == True:
            img = np.concatenate((img,img2),axis=1)
        if self.out is not None:
            self.out.write(img.astype(np.uint8))
        if self.split == True:
            cv2.imshow('image',cv2.resize(img, (2000,1000), interpolation = cv2.INTER_AREA))
        
        else:
            cv2.imshow('image',cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            exit(0)
        sleep(1/30)
        return True
    """
    def __del__(self):
        if self.filename is not None:
            print("release")
            self.out.release()
            cv2.imwrite(self.filename+"_trajectories_THICK.png",self.img2show)
        cv2.destroyAllWindows()



def sanityChecks():
    from datetime import datetime
    DEFAULT_PATH = "SanityChecks"
    if not os.path.exists(DEFAULT_PATH):
        os.makedirs(DEFAULT_PATH)
    function = Positive_weighted_distances()
    function = BoidsMetrisch()
    dsize=(1500,1500)
    pos1 = [[dsize[0]//2,dsize[1]//2+(i*50)] for i in range(10)]
    pos2 = [[dsize[0]//2+1,dsize[1]//2+(i*50)] for i in range(10)]
    initialPositions = [pos1,pos2]
    now = datetime.now()
    name = "Parallel_"+"no_attraction_"
    generator = GenerateFakeData(function,
                                #randomRange=[(3.0,3.0),(1,1),(1.0,1.0),(0.0,0.0)],
                                randomRange=[(50,50),(70,70),(150,150),(90,90),(360,360)],
                                randomParameter=False,
                                initialPositions=initialPositions,
                                nbr_fish = 10,std=50,dsize=dsize,
                                filename = os.path.join(DEFAULT_PATH,name+str(function)+"_"+now.strftime("%m-%d-%Y-%H-%M-%S")))
    for i in range(100000):
        print(i)
        generator()

    cv2.destroyAllWindows()
        

def main():
    #function = Positive_weighted_distances()
    #function = moveroutine_boids_MultiFish
    #function = moveroutine_boids
    #randomRange=[(0,100),(100,200),(200,300),(10,20),(0,1)]
    #generator = GenerateFakeData(function,
    #    nbr_fish=10,
    #    randomRange=[(0,2),(0,2),(0,2),(0,2)],
    #    std=500,
    #    randomParameter=True)
    function = BoidsMetrisch()
    function = Positive_weighted_distances()
    

    generator = GenerateFakeData(function,
                                randomRange=[(5.5,5.5),(1.0,1.0),(1.0,1.0),(1.0,1.0)],
                                #randomRange=[(250,250),(255,255),(200,200),(90,90),(360,360)],
                                #randomRange=[(50,100),(150,200),(200,500),(2,5),(0.2,0.2)],
                                randomParameter=False,
                                nbr_fish = 20,std=100,dsize=(2000,2000))
    for i in range(100000):
        print(i)
        generator()

    cv2.destroyAllWindows()
    
    

if __name__ == '__main__':
    main()