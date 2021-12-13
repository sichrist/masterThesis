import numpy as np
import cv2
import time
import os


PATH   = "/mnt/HDD-1/Backup2021/Dokumente/collective_behaviour/Data/TransitioningBehaviourInSchoolingFish"

# 30 Fish (golden shiners )
Video1 = "pcbi.1002915.s013.m4v"
# 70 Fish (golden shiners )
Video2 = "pcbi.1002915.s014.m4v"
# 150 Fish (golden shiners )
Video3 = "pcbi.1002915.s015.m4v"
# 300 Fish (golden shiners )
Video4 = "pcbi.1002915.s016.m4v"

nbr_of_fish = {Video1: 30, Video2: 70, Video3: 150, Video4: 300}

mask = cv2.imread('mask.png')
ids = np.where(mask == [0,255,0])



video = Video1

mean_img = np.zeros(mask.shape)
imgstack = []
vidcap = 0
vidcap = cv2.VideoCapture(os.path.join(PATH,video))
ctr = 0
success,image = vidcap.read()
while success:
    success,image = vidcap.read()
    if success: 
        mean_img += cv2.GaussianBlur(image,(5,5),0)
        imgstack.append(cv2.GaussianBlur(image,(5,5),0))
        ctr +=1
if vidcap:
    vidcap.release()


std_img  = np.std(imgstack)
mean_img = mean_img // ctr

def get_mean_img():

    video = Video1

    mean_img = np.zeros(mask.shape)
    imgstack = []
    vidcap = 0
    vidcap = cv2.VideoCapture(os.path.join(PATH,video))
    ctr = 0
    success,image = vidcap.read()
    while success:
        success,image = vidcap.read()
        if success: 
            mean_img += cv2.GaussianBlur(image,(5,5),0)
            #mean_img +=
            imgstack.append(cv2.GaussianBlur(image,(5,5),0))
            ctr +=1
    if vidcap:
        vidcap.release()


    std_img  = np.std(imgstack)
    mean_img = mean_img // ctr
    #mean_img = np.mean(imgstack,axis=(1,2))


    return mean_img.copy()


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


def to_binary(img, threshold=100):
    
    foreground = np.where(img >= threshold)
    x = foreground[0]
    y = foreground[1]
    im = np.zeros(img.shape)
    im[x,y] = 1
    return im


def remove_borders(img,to=[255,255,255]):
    img[ids[0],ids[1]] = to
    return img


def diff_imgs(img0,img1):
    img = img1 - img0
    img[np.where(img < 0)] = 0
    return img
    


def RegionLabeling(I):
    label = 2
    coords = np.where(I == 1)

    for x,y in zip(coords[0],coords[1]):
        found,I = FloodFill(I,x,y,label)
        if found:
            label += 1
    return label,I


def remove_mean(img):
    return mean_img - img

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


def colorLabel(I,label):
    img = np.zeros((*I.shape,3))
    
    for i in range(2,label):
        ids = np.where(I == i)
        c = rdmclr()
        img[ids[0],ids[1]] = c
        
    return img


def toGray(img):
    if len(img.shape) <= 2:
        return img
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.144 * img[:,:,2]
    return gray


def label_Img(img,threshold=50,preprocess=True):
    if preprocess:
        kernel = np.ones((2,2),np.uint8)
        img = remove_mean(img)
        img = remove_borders(img,to = 0)
        img = cv2.dilate(img,kernel,iterations = 1)
    img = (img/img.max())*255
    img.astype(np.uint8)
    img = toGray(img)
    img = to_binary(img,threshold)
    
    
    
    label,img = RegionLabeling(img)

    return (label,img)


def com(idx):
    x = idx[0].mean()
    y = idx[1].mean()
    return int(x),int(y)


def fishes_to_coords(img,labels,original):
    coords = []
    texture = []
    for i in range(2,labels):
        ids = np.where(img == i)
        coords.append(ids)
        texture.append(original[ids])
    return coords,texture

def extract_fish(video,threshold,show_video=True,additionalText = None,start_with_frame=134,texture = False,path=PATH,preprocess=True):
    vidcap = 0
    if vidcap:
        vidcap.release()

    vidcap = cv2.VideoCapture(os.path.join(path,video))
    success,image = vidcap.read()
    count = 0

    fish = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_min = 5000
    n_max = 0
    frames = 0
    labels_per_frame = []
    nbr_labels_total = 0
    extracted_fish = []
    texture_fish = []



    while success:
        success,image = vidcap.read()
        if success:
            count += 1
            if count < start_with_frame:
                continue
            frames += 1
            original = image


            labels,image = label_Img(image,threshold=threshold,preprocess=preprocess)
            labels_per_frame.append(labels)

            coords,extracted_texture = fishes_to_coords(image,labels,original)
            extracted_fish.append(coords)
            if texture:
                texture_fish.append(extracted_texture)


            for i in range(2,labels):
                x,y = com(np.where(image == i))
                original = cv2.circle(original, (y,x), 3, [0,255,0], 2)
                original[x,y] = [255,255,255]

            idcs = np.where(image>0)
            image[idcs] = 255
            image = np.dstack([image]*3)

            if labels > n_max:
                n_max = labels-1
            if labels < n_min:
                n_min = labels-1
            nbr_labels_total += labels -1
            
            if show_video:
                cv2.putText(image,str(labels)+"/"+str(nbr_of_fish[video]),(25,25), font, .5,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(image,str(n_min)+"/"+str(n_max),(25,50), font, .5,(255,0,0),1,cv2.LINE_AA)
                cv2.putText(image,str(nbr_labels_total//frames),(25,75), font, .5,(0,0,255),2,cv2.LINE_AA)
                if additionalText is not None:
                    cv2.putText(image,additionalText,(25,100), font, .5,(0,255,255),1,cv2.LINE_AA)


                original = np.concatenate((original,image),axis=1)
                show_image(original)
                
    return {"labels found":labels_per_frame,
            "threshold":threshold,
            "extracted fish":extracted_fish,
            "texture": texture_fish}


class ImgViewer(object):
    """docstring for ImgViewer"""
    def __init__(self,fps=30):
        super(ImgViewer, self).__init__()
        self.fps = fps
        self.timestep = 1.0/self.fps
        self.windowname = 'ImgViewerFrame'
        cv2.namedWindow(self.windowname)
        cv2.moveWindow(self.windowname,2600,40)

        self.time_started = None



    def __call__(self,img):
        cv2.imshow(self.windowname,img)
        time.sleep(self.timestep)


class VideoStream:
    def __init__(self,video,skip = 134,gray=True,invert = False,path = PATH):
        self.path = os.path.join(path,video)
        self.video = video
        self.vidcap = None
        self.skip = skip
        self.ctr = 0
        self.gray = gray
        self.invert = invert
        self.mean = None
        self.min = None
        self.max = None
        

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
            if self.gray:
                image = toGray(image)
            if self.invert:
                image = 255 - image
                
            return image
        
        else:
            self.ctr = 0
            if self.vidcap:
                self.vidcap.release()
            self.vidcap = None
            return None
        
        
    def getMeanImg(self):
        if self.mean is not None:
            return self.mean
        
        img = self()
        newimg = img
        if self.min is None:
            self.min = 255
            self.max = 0
        
        
        ctr = 0
        while True:
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
            




