import cv2
import numpy as np
import os

"""

    A collection of Tracking functions/classes of this directory
    All files except this one should be obsolete

"""
def draw_flow(imgs,flow, step=4):

    h, w = imgs.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)

    lines = np.int32(lines + 0.5)
    #vis = img.copy()
    vis = np.zeros_like(imgs)
    #vis = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (255, 255, 255))
    #for (x1, y1), (x2, y2) in lines:
    #    cv.circle(vis, (x1, y1), 1, (255, 255, 0), -1)
    return vis

def calcFlow(prvs,nxt,flow=None,windowsize=5,polyexp=5):
    return cv2.calcOpticalFlowFarneback(prvs,nxt, 
                                        flow, 
                                        0.5, # pyramid-scale 0.5 = classical pyr.. each layer 0.5 smaller
                                        4,   # levels of pyramid
                                        windowsize,  # windowsize > 4 robustness
                                        4,   # iteration @ each lvl
                                        polyexp,   # size of the pixel neighborhood used to find polynomial expansion in each pixel; 
                                             # larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
                                        1.2, 
                                        0)

def com(idx):
    x = idx[0].mean()
    y = idx[1].mean()
    return int(x),int(y)

def resize(img,scale_percent = 20):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img,dim)

def show_image(a, fmt='jpeg'):
    ##
    # Display Image-stream in ipython
    # Does not work smoothly in firefox
    #
    
    import numpy as np
    from IPython.display import clear_output, Image, display
    from io import StringIO, BytesIO
    import PIL.Image
    from time import sleep
    
    faux_file = BytesIO()
    
    a = np.uint8(np.clip(a, 0, 255))
    PIL.Image.fromarray(a).save(faux_file, fmt)
    clear_output(wait=True)
    imgdata = Image(data=faux_file.getvalue())
    display(imgdata)

def toGray(img):
    if len(img.shape) <= 2:
        return img
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.144 * img[:,:,2]
    return gray

def saveFish(path2file,coords):
    file = open(path2file,"a")
    line2save = ""
    ctr = 0
    for i,fish in enumerate(coords):
        ctr += 1
        line2save += str(fish[0])+" | "+str(fish[1])
        if i<len(coords) -1:
            line2save+=","
    
    line2save += ","+str(coords[-1][0])+" | "+str(coords[-1][1])
    ctr += 1
    if len(coords[-1][0]) != len(coords[-1][1]):
        print("MASSIVE ERROR")
        raise AttributeError
        

    line2save = line2save.replace("\n","")
    line2save+="\n"
    file.write(line2save)
    file.close()

def readline(filename):

    file = open(filename, 'r')
    count = 0

    def str2nparray(array):
        array = array.replace("[","")
        array = array.replace("]","")
        array = array.replace('\n','')
        array = array.split(" ")
        array.remove('')
        #print(array,len(array))
        array = [int(a) for a in array if a.isdecimal()]
        return np.array(array)


    while True:
        count += 1
        line = file.readline()
        if line:
            f = line.split(",")
            #print(f)
            fish = []
            for a in f:
                #print(a)
                x,y = a.split("|")             
                fish.append((str2nparray(x),str2nparray(y)))
            yield fish
        else:
            return

def objects_to_coords(img,labels):
    coords = []
    texture = []
    for i in range(2,labels+1):
        ids = np.where(img == i)
        coords.append(ids)

    return coords

def RegionLabeling(I):
    label = 2
    coords = np.where(I == 1)

    for x,y in zip(coords[0],coords[1]):
        found,I = FloodFill(I,x,y,label)
        if found:
            label += 1
    return label,I

def FloodFill(I,u,v,label):
    from collections import deque
    x_,y_ = I.shape[:2]
    
    stack = deque()
    stack.append((u,v))
        
    found = False
    
    while len(stack) > 0:
        
        x,y = stack.pop()
        

        if x>=0 and x < x_ and y >= 0 and y < y_ and I[x,y] == 1:
            found = True
            I[x,y] = label
            stack.append((x+1,y))
            stack.append((x,y+1))
            stack.append((x-1,y))
            stack.append((x,y-1))
            
            # 8er nachbarscharft
            stack.append((x+1,y+1))
            stack.append((x-1,y-1))
            stack.append((x-1,y+1))
            stack.append((x+1,y-1))
        
    return found,I

def to_binary(img, threshold=100):
    
    foreground = np.where(img >= threshold)
    x = foreground[0]
    y = foreground[1]
    im = np.zeros(img.shape)
    im[x,y] = 1
    return im

class NN_Tracker(object):

    """

        This class is the main tracking-class

    """

    def __init__(self, path2video,Threshold=50,Datafolder = "data",skipFrames = 0):
        super(NN_Tracker, self).__init__()
        self.path2video = path2video
        self.Threshold  = Threshold
        self.skipFrames = skipFrames
        self.Datafolder = os.path.join(os.path.dirname(path2video),Datafolder)
        self.path2trajectories = os.path.join(self.Datafolder,os.path.basename(path2video).split(".")[0] + ".txt")
        if not os.path.exists(self.Datafolder):
            os.mkdir(self.Datafolder)

    def extract_Objects(self,threads = 1):
        np.set_printoptions(threshold=np.inf)
        framesToSkip = 0
        try:
            for i,line in enumerate(readline(self.path2trajectories)):
                framesToSkip = i
                print("{:07d} Lines found".format(framesToSkip),end="\r")
            framesToSkip -= 1
        except:
            pass
        video    = VideoStreamer(self.path2video,skip = framesToSkip)
        video.openVideo()
        img = video()
        meanimg = self.getMeanImg()
        meanimg = meanimg.astype(np.float64)
        while img is not None:
            print("Extracting Frame: {:07d}".format(framesToSkip),end="\r")
            framesToSkip += 1
            img = img.astype(np.float64)
            img = meanimg - img
            img[np.where(img < 0)] = 0
            img = (img/img.max())*255
            img.astype(np.uint8)
            img = toGray(img)
            img = to_binary(img,self.Threshold)
            label,I = RegionLabeling(img)
            coords  = objects_to_coords(I,label)
            saveFish(self.path2trajectories,coords)
            img = video()

    def getPath2Trajectories(self):
        return self.path2trajectories

    def getMeanImg(self,frames = 2000):
        filename     = os.path.basename(self.path2video).split(".")[0]+".png"
        file_meanImg = os.path.join(self.Datafolder,filename)
        if os.path.exists(file_meanImg):
            return cv2.imread(file_meanImg)


        video    = VideoStreamer(self.path2video,skip = self.skipFrames)
        video.openVideo()
        mean_img = video()
        mean_img = mean_img.astype(np.float64)
        ctr      = 1
        for i in range(frames):
            img  = video()
            if img is None:
                break
            mean_img += img
            ctr += 1


        mean_img = mean_img/ctr
        mean_img = mean_img.astype(np.uint8)
        cv2.imwrite(file_meanImg,mean_img)


        return mean_img

    def show_extracted_objects(self,min_size = 0,imgtransform = []):
        video    = VideoStreamer(self.path2video)
        video.openVideo()
        for i,line in enumerate(readline(self.path2trajectories)):
            img = video()
            if img is None:
                break
            for f in line:
                if len(f[0]) < min_size:
                    continue
                img[f[0].astype(np.int),f[1].astype(np.int),:] = [255,0,0]
            for f in imgtransform:
                img = f(img)
            show_image(img)

    def getFish_sequentially(self,texture=False):
        if texture:
            video    = VideoStreamer(self.path2video)
            video.openVideo()


        for line in readline(self.path2trajectories):
            if texture:
                img = video()
                txtre = []
                for fish in line:
                    if len(fish[0]) == 0:
                        t = (None)
                    else:
                        t = (img[fish])
                    txtre.append(t)
                yield line, txtre
            else:
                yield line

    def get_random_fish_img(self,fishperimg=10,buffersize=1024,imgshape=(512,512),size=(None,None),maximages=10000):

        fishlist = []
        fish_generator = self.getFish_sequentially(texture = True)
        meanIMG = self.getMeanImg()
        imgctr = 0
        while imgctr < maximages:
            fish,texture = next(fish_generator)
            pic = np.zeros(imgshape)
            label = np.zeros_like(pic)
            pic = np.dstack([pic]*3)
            
            background_x = np.random.randint(low=0, high=meanIMG.shape[0] - imgshape[0], size=None, dtype=int)
            background_y = np.random.randint(low=0, high=meanIMG.shape[1] - imgshape[1], size=None, dtype=int)
            pic = meanIMG[background_x:background_x+imgshape[0],background_y:background_y+imgshape[1],:].copy()
            for i,f in enumerate(fish):

                if size[0] is not None and size[1] is not None:
                    if len(f[0]) < size[0] or len(f[0]) > size[1]:
                            continue
                x_min,x_max = f[0].min(),f[0].max()
                y_min,y_max = f[1].min(),f[1].max()
                diff_x      = np.abs(x_min-x_max)
                diff_y      = np.abs(y_min-y_max)
                x           = np.random.randint(low=diff_x, high=imgshape[0]-diff_x, size=None, dtype=int)
                y           = np.random.randint(low=diff_y, high=imgshape[1]-diff_y, size=None, dtype=int)
                newpos = f[0]-x_max+x,f[1]-y_max+y
                pic[newpos] = texture[i]
                label[com(newpos)] = 1
                
            imgctr+=1
            yield pic,label

    def calcTrajectories(self):
        video    = VideoStreamer(self.path2video)
        video.openVideo()
        nxt = video()
        nxt = toGray(nxt)
        while nxt is not None:

            prvs = nxt
            nxt  = video() 
            nxt  = toGray(nxt)

            flow = calcFlow(prvs,nxt)
            vis = draw_flow(prvs,flow)
            show_image(vis)





class VideoStreamer(object):
    """docstring for VideoStreamer"""
    def __init__(self, path2video, skip = 0):
        super(VideoStreamer, self).__init__()
        self.path2video = path2video
        self.skip       = skip
        self.vidcap     = None

    def openVideo(self):
        self.ctr = 0
        self.vidcap = cv2.VideoCapture(self.path2video)
        
    def __call__(self):
        
        if self.vidcap is None:
            self.openVideo()
        
        while self.ctr < self.skip:
            success,image = self.vidcap.read()
            self.ctr += 1
            
        success,image = self.vidcap.read()

        if success:
            return image

        else:
            self.ctr = 0
            if self.vidcap:
                self.vidcap.release()
            self.vidcap = None
            return None
        
