import numpy as np
from .Fish import *

def com(idx):
    x = idx[0].mean()
    y = idx[1].mean()
    if np.isnan(x) or np.isnan(y):
        return None,None
    return int(x),int(y)

def toGray(img):

    """
        We convert the image to grayscale, since we dont need color
    """

    if len(img.shape) <= 2:
        return img
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.144 * img[:,:,2]
    return gray.astype(np.uint8)

def to_binary(img, threshold=100):

    """
        transfer image in binary format
        I[x,y] >= threshold => 1, 0 otherwise
    """

    foreground = np.where(img >= threshold)
    x = foreground[0]
    y = foreground[1]
    im = np.zeros(img.shape)
    im[x,y] = 1
    return im

def FloodFill(I,u,v,label):

    """
        This is for extracting the individuals
    """

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

def RegionLabeling(I):

    """
        This is for extracting all individuals from one image
        The image needs to be in binary format
    """

    label  = 2
    coords = np.where(I == 1)

    for x,y in zip(coords[0],coords[1]):
        found,I = FloodFill(I,x,y,label)
        if found:
            label += 1
    return label,I

def label2Fish(i,img):
    idx = np.where(img == i)
    if len(idx[0]) == 0:
        return None
    return Fish(idx,texture=img[idx],id=i)

def extract(video,threshold=10):
    mean_img = video.getMeanImg()

    img = video()
    mean_img = video.getMeanImg()
    while img is not None:
        img                   -= mean_img
        img[np.where(img < 0)] = 0
        img                    = toGray(img)
        img                    = to_binary(img,threshold)
        label,I                = RegionLabeling(img)
        BlobList               = [label2Fish(i,I) for i in range(2,label) ]
        yield BlobList
        img                    = video()



