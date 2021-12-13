import numpy as np
import cv2
import os
from Utils.Common import l2error
class VideoStream:
    def __init__(self,video,skip = 0,gray=True,invert = False):
        self.path = video
        self.video = video
        self.vidcap = None
        self.skip = skip
        self.ctr = 0
        self.gray = gray
        self.invert = invert
        self.mean = None
        self.min = None
        self.max = None
        self.op  = []

    def openVideo(self):
        self.vidcap = cv2.VideoCapture(self.path)

    def __call__(self):
        
        if self.vidcap is None:
            self.openVideo()
        
        while self.ctr < self.skip:
            success,image = self.vidcap.read()
            self.ctr += 1
            
        success,image = self.vidcap.read()
        if success:
            for op in self.op:
                image = op(image)
            return image
        
        else:
            self.ctr = 0
            if self.vidcap:
                self.vidcap.release()
            self.vidcap = None
            return None
        
    def addOperation(self,op):
        self.op.append(op)
        
    def getMeanImg(self,max_frames = 2000):
        if self.mean is not None:
            return self.mean
        
        img = self()
        img = np.zeros_like(img,dtype=np.uint32)
        newimg = img
        if self.min is None:
            self.min = 255
            self.max = 0
        
        
        ctr = 0
        while ctr<max_frames:
            print("{:05d}".format(ctr),end="\r")
            newimg = self()
            if newimg is None:
                break
            img += newimg
            ctr += 1
            if newimg.min() < self.min:
                self.min = newimg.min()
            if newimg.max() > self.max:
                self.max = newimg.max()
        self.mean = img / ctr
        return self.mean
    
    
    def minima(self):
        if self.min is None:
            self.getMeanImg()
        return self.min
    
    
    def maxima(self):
        if self.max is None:
            self.getMeanImg()
        return self.max

def sim(pos_atm,pos_last,x,function):
    y = []
    for i in range(len(pos_atm)):
        y.append(function(pos_atm,pos_last,*x[i],i))
    y = np.array(y)
    return np.squeeze(y)

 
def resize(img,scale_percent = 20):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img,dim)

def SaveAsVideo(realdata,
                video_path,
                fish,
                filename,
                function,
                dsize = (2160,2160)):
    video = VideoStream(video_path,skip=2,gray=False)
    img = video()
    frame = 0
    def dots2frame(img,tra,size=10,color = (255,255,255)):
        for fid,a in enumerate(tra):
            a = np.nan_to_num(a)
            x,y = int(a[0]),int(a[1])
            cv2.circle(img,(x,y),size,color,-1)
        return img

    def path2img(img,pos,color=(255,255,255),maxlen=10):
        if len(pos)<2:
            return img
        
        for i in range(len(pos)-1,len(pos)-maxlen,-1):
            if i <0:
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

    font = cv2.FONT_HERSHEY_SIMPLEX
    def error2img(img,pos_pred,pos_true):
        for a,b in zip(pos_pred,pos_true):
            error = "{:.2e}".format(l2error(a,b))
            x,y = int(a[0]),int(a[1])
            cv2.putText(img,error,(x,y-150), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.line(img,(x,y),(x,y-150),(255,0,255),2)
        return img

    true_path = []
    pred_path = []
    frame_list = []
    while img is not None:
        #img = dots2frame(img,realdata[frame])
        pos_atm = realdata[frame-1]
        pos_last = realdata[frame-2]
        pos_true = realdata[frame]

        img = dots2frame(img,pos_true,size=10)
        pos_pred = sim(pos_atm,pos_last,fish[:,frame,:-1],function)
        img = dots2frame(img,pos_pred,size=10,color=(255,255,0))
        true_path.append(pos_true)
        pred_path.append(pos_pred)


        img = path2img(img,true_path[1:])
        img = path2img(img,pred_path[1:],color=(255,255,0))
        img = error2img(img,pos_pred,pos_true)
        
        #show_in_Notebook(img)
        frame_list.append(img)
        img = video()
        frame+=1

        if frame == fish.shape[1]:
            break

    size = size = frame_list[0].shape[:2]  
    tmppath = ".tmp"
    os.mkdir(tmppath)
    print(tmppath)
    for i in range(len(frame_list)):
        cv2.imwrite(os.path.join(tmppath,'img'+str(i)+".png"), frame_list[i])
    s = len(frame_list)
    del frame_list
    out = cv2.VideoWriter(filename+".avi",cv2.VideoWriter_fourcc(*'FMP4'), 15, dsize)

    for i in range(s):
        img = cv2.imread(os.path.join(tmppath,'img'+str(i)+".png"))
        output = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
        out.write(output)
        del img
        del output
    out.release()
    os.rmdir(tmppath)

def saveAsVideo_(pred_pos,
                true_pos,
                video_path,
                filename,
                offset = 2,
                dsize = (2160,2160),
                show=False):
    from time import sleep
    video = VideoStream(video_path,skip=offset,gray=False)
    img = video()

    def dots2frame(img,tra,size=10,color = (255,255,255)):
        for fid,a in enumerate(tra):
            a = np.nan_to_num(a)
            x,y = int(a[0]),int(a[1])
            cv2.circle(img,(x,y),size,color,-1)
        return img

    def path2img(img,pos,color=(255,255,255),maxlen=10):
        if len(pos)<2:
            return img
        
        for i in range(len(pos)-1,len(pos)-maxlen,-1):
            if i <0:
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

    frame = 0
    out = cv2.VideoWriter(filename+".avi",cv2.VideoWriter_fourcc(*'FMP4'), 15, dsize)

    while img is not None:
        img = dots2frame(img,true_pos[frame])
        img = dots2frame(img,pred_pos[frame],size=8,color=(255,255,0))
        img = path2img(img,pred_pos[:frame],color=(255,255,0))
        frame += 1
        img = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
        out.write(img)
        if show:
            cv2.imshow('image',cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            sleep(0.5)

        img = video()
        if frame >= len(pred_pos):
            break
    out.release()
    cv2.destroyAllWindows()