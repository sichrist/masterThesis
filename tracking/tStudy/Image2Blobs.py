import sys
import os
import cv2
from Tracking import VideoStreamer,NN_Tracker, resize, com, show_image,calcFlow,toGray
from .AI_Fish import SchoolOfFishStream_AI
import traceback
import numpy as np
import cupy as cp
from multiprocessing import Process, Queue, Lock, Semaphore, Value, Condition
from time import time
import cupyx.scipy.ndimage as cpyx
import scipy
import re
from .Blobs import *


windowname = 'OpenCvFrame'
MAX_W      = 2540
MAX_H      = 2540

def toGray(img):
    if len(img.shape) <= 2:
        return img
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.144 * img[:,:,2]
    return gray

def to_binary(img, threshold=20):
    
    foreground = cp.where(img >= threshold)
    x = foreground[0]
    y = foreground[1]
    im = cp.zeros(img.shape)
    im[x,y] = 1
    return im

def consumer(queue,blobqueue,bloblock,lock,actual_frame,threshold,frame_condition):
    from time import sleep
    def process(condition,lock):
        #with lock:
            #print("W8 next frame ",condition)

        condition.acquire()
        condition.wait()
        condition.release()

    
    """
        Queue     -> 4 slice of image
        Blobqueue -> 4 blobs from slices
        
    """
    try:
        while True:
            frame,offset,total,nbr,img = queue.get()
            #with lock:
                #print("Consumer got frame: ",frame," slicenbr: ",nbr)
            h,w = img.shape[:2]
            blobs = []

            lines = [(i+offset,[]) for i in range(img.shape[0])]

            start = time()
            idx = np.where(img == 1)
            y  = None
            xs = None
            xe = None       
            for y_,x_ in zip(idx[0],idx[1]):


                if y is not None:
                    if y == y_ and xe + 1 == x_:
                        xe = x_
                    else:
                        #blobs.append((xs,xe,y+offset))
                        lines[y][1].append((xs,xe,y+offset) )
                        y  = None
                        xs = None
                        xe = None 

                if y is None:
                    y = y_
                    xs = x_
                    xe = x_

            if y is not None:
                lines[y][1].append((xs,xe,y+offset) )


            while actual_frame.value != frame:
                process(frame_condition,lock) 

            blobqueue.put((frame,offset,total,nbr,lines))

                
    except KeyboardInterrupt:
        print("FIN")

def producer(queue,viewerQeueu,processes,path2Video,lock,use_sobel = False,use_closing = True):
    
    if type(path2Video) is SchoolOfFishStream_AI:
        video = path2Video
        mean_img = video.getMeanImg()
        mean_img = cp.array(mean_img,dtype=cp.float64)
    else:
        mean_img = NN_Tracker(path2Video).getMeanImg()
        video = VideoStreamer(path2Video)
        mean_img = cp.array(mean_img,dtype=cp.float64)
        
    img = video()
    slices = processes
    x,y = img.shape[:2]
    steps = x//slices
    shape = [[0,1,0],[1,1,1],[0,1,0]]
    frame = 0
    try: 
        while img is not None:


            viewerQeueu.put((frame,img))
            start = time()
            img = cp.array(img,dtype=cp.float64)
            img = img.astype(np.float64)
            if mean_img.mean() > 0:
                img = mean_img - img
            else:
                img = img - mean_img
            img = toGray(img)
            if use_closing:
                img = cpyx.grey_closing(img,size=(5,5))
            if use_sobel:
                sx = cpyx.sobel(img,axis=0,mode='constant')
                sy = cpyx.sobel(img,axis=1,mode='constant')
                sobel=cp.hypot(sx,sy)
                sobel = ((sobel-sobel.min())*255/sobel.max()).astype(cp.uint8)
                img  = sobel - img
                img[cp.where(img < 0 )] = 0
                cv2.imshow(windowname,sobel.get())
                if cv2.waitKey(25) & 0XFF == ord('q'):
                    raise KeyboardInterrupt

            img = to_binary(img)

            np_img = img.get().astype(np.uint8)
            del img
            cp._default_memory_pool.free_all_blocks()
            for nbr,i in enumerate(range(0,x,steps)):
                queue.put((frame,i,slices,nbr,np_img[i:i+steps,:]))

            while queue.full(): 
                continue

            img = video()
            frame += 1
    
        print("NO IMGS LEEEFFFT")

    except KeyboardInterrupt:
        del img
        del mean_img
        cp._default_memory_pool.free_all_blocks()
    del mean_img
    cp._default_memory_pool.free_all_blocks()

    # wait for Viewer to consume all images
    while viewerQeueu.full():
        continue

def viewer(blobqueue,bloblock,viewerQeueu,nProcesses,actual_frame,frame_condition,lock,blobfunction,returnvals,finflag):
    
    frame_needed = 0
    from time import sleep
    def resize(img,scale_percent = 20):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img,dim)
    
    def wake_up(condition,lock):

        condition.acquire()
        condition.notify_all()
        condition.release()
                


    slices = 0
    blobs = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = [0]
    img_frame = None
    blobsFound = []

    retblobs = []
    try:
        
        while True:


            with actual_frame.get_lock():
                actual_frame.value = frame_needed

            wake_up(frame_condition,lock)
            start = time()
            try:
                frame,offset,total,nbr,blob = blobqueue.get(block=False,timeout=0.1)
            except :
                continue
            blobs.append(blob)
            slices += 1

            msg.append(frame)

            if slices == total:

                slices = 0

                img_frame,img = viewerQeueu.get()
                returnvals.put(blobfunction(blobs,img))
                #retblobs.append(blobfunction(blobs,img))
                msg = [frame_needed]
                
                blobs = []

                frame_needed += 1
                img_frame = None

            with finflag.get_lock():
                if finflag.value == 1:
                    break
    except KeyboardInterrupt:
        print("FINito")



    #print("put")
    #returnvals.put(retblobs)
    #sleep(5)

    
class LinkedBlobs(object):
    """docstring for LinkedBlobs"""
    def __init__(self):
        super(LinkedBlobs, self).__init__()
        self.BlobList = {}

    def addBlob(self,blob,trajectories):
        for i,(x,y) in enumerate(trajectories):
            if blob.insideBlob(x,y):
                blob.addCenter(x,y)
                if i not in self.BlobList:
                    self.BlobList[i] = [((x,y),blob)]
                else:
                    self.BlobList[i].append(((x,y),blob))

    def __call__(self,last=5):
        ret_list = []
        for key in self.BlobList:
            ret_list.append((key,self.BlobList[key][-last:]))
        return ret_list
      
BLOB_HISTORY  = LinkedBlobs()

def slices2blobs(slices,img):


    from operator import itemgetter

    blobs        = []
    active_blobs = []
    

    slices = [x for sublist in slices for x in sublist]
    slices = sorted(slices,key = lambda tup:tup[0])

    # assign a new blob to all Lines of the first row
    active_blobs = [Blob(line) for line in slices[0][1]]
    
    
    LastRow = slices[0][1]
    
    def findBlob(line):
        for i,blob in enumerate(active_blobs):
            if blob.partOfMe(line):
                return blob
        return None

    
    for idx,Row in slices[1:]:
        j = 0
        k = 0
        
        # if no lines exist, there will be no active blobs
        if len(Row) == 0:
            blobs += active_blobs
            active_blobs = []
            LastRow = Row    
            continue
            
        if len(LastRow) == 0:
            active_blobs = [Blob(line) for line in Row]
            LastRow = Row 
            continue
            
        while j < len(Row):

            
            xs0,xe0,y0 = Row[j]
            if k<len(LastRow):
                xs1,xe1,y1 = LastRow[k]
            else:
                k=-1
                
            # if no lines in LastRow left
            # ........................  xs1,xe1,y1
            #      |:::|      |::| |::| xs0,xe0,y0
            #
            if k < 0:
                active_blobs.append(Blob((xs0,xe0,y0)))
                j += 1
                continue
                
                

            # if Lines are further apart (y) , create new blob 
            #        |:::|         xs1,xe1,y1
            # ....................
            # |:::|                xs0,xe0,y0
            if y1+1 != y0:
                active_blobs.append(Blob((xs0,xe0,y0)))
                j += 1
                continue
            
            # if Line ends before in prev Line starts
            #        |:::|         xs1,xe1,y1
            # |:::|                xs0,xe0,y0
            if xe0 < xs1+1:
                active_blobs.append(Blob((xs0,xe0,y0)))
                j+=1
                continue
            
            
            # if line in prev ends before Line starts
            # |:::|                xs1,xe1,y1
            #       |:::|          xs0,xe0,y0
            #
            elif xe1+1 < xs0:
                k+=1
                continue
                
            # last case, they intersect
            # |:::|    |:::|       xs1,xe1,y1
            #   |:::| |:::|        xs0,xe0,y0
            elif not (xe0+1 < xs1 or xe1+1 < xs0):
                blob       = findBlob((xs1,xe1,y1))
                blob2merge = findBlob((xs0,xe0,y0)) 

                
                if blob2merge is None:
                    blob.add((xs0,xe0,y0))
                else:
                    blob.merge(blob2merge)
                    active_blobs.remove(blob2merge)
                nb = findBlob((xs0,xe0,y0))
                if nb is None:
                    return
                    #exit(-1)
                    
                if xe1 < xs0:
                    k += 1
                else:
                    j += 1
                    

        
        LastRow = Row        
                    
    blobs += active_blobs

    mi = 100000
    ma = 0
    
    def random_color():
        import random
        rgbl=[255,0,0]
        random.shuffle(rgbl)
        return tuple(rgbl)

    idx = 0
    nbr_blobs = 0



    for i,blob in enumerate(blobs):

        nbr_blobs +=1

        for x0,x1,y in blob.lines:
            img[y,x0:x1,:] = [255,0,0]

        if len(blob) > ma:
            ma = len(blob)
            idx = i        
                       
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(ma), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(nbr_blobs), (20,550), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

    h,w = img.shape[:2]

    if h > MAX_H or w > MAX_W:
        img = resize(img,scale_percent=30)


    cv2.imshow(windowname,img)
    if cv2.waitKey(25) & 0XFF == ord('q'):
        raise KeyboardInterrupt

    if cv2.waitKey(25) & 0XFF == ord(" "):
        print("Stop")
        sleep(5)
        while cv2.waitKey(25) & 0XFF != ord("m"): 
            continue
        print("Continue")

    return blobs



def start(fish_video,threshold = 30,processes = 20,blobfunction=slices2blobs,use_sobel = False,use_closing = True):

    QueueSize = 150
    queue = Queue(maxsize=QueueSize)
    blobqueue = Queue(maxsize=5000)
    lock  = Lock()
    returnvals = Queue()

    # lock for consuming blobs
    blobLock        = Semaphore()
    actual_frame    = Value('i',0)
    finflag         = Value('i',0)
    consumers       = []
    
    frame_condition=Condition(Lock())

    viewerQeueu = Queue()

    # start viewer first
    view = Process(target=viewer, args=(blobqueue, blobLock,viewerQeueu, processes,actual_frame,frame_condition,lock,blobfunction,returnvals,finflag))
    view.daemon = True
    view.start()


    for i in range(processes):
        p = Process(target=consumer, args=(queue, blobqueue,blobLock,lock,actual_frame,threshold,frame_condition))
        p.daemon = True
        consumers.append(p)


    prod = Process(target=producer, args=(queue, viewerQeueu,processes,fish_video,lock,use_sobel,use_closing))
    #prod.daemon = True
    prod.start()



    for c in consumers:
        c.start()

    try:
        prod.join()


    except KeyboardInterrupt:
        traceback.print_exc()
        for c in consumers:
            c.terminate()
        prod.terminate()
        view.terminate()
        cv2.destroyAllWindows()
        consumers = None

    retlist = []
    with finflag.get_lock():
        finflag.value = 1

    while returnvals.qsize() > 0:
        val = returnvals.get()
        retlist.append(val)
        
    if consumers is not None:
        for c in consumers:
            c.terminate()
        prod.terminate()
        view.terminate()
        cv2.destroyAllWindows()
    
    return retlist


def blobs2File(allblobs,path2file):
    def lines2string(line):
        retstr = "("

        for l in line:
            retstr += str(l[0])+","+str(l[1])+","+str(l[2])+","
        retstr = retstr[:-1]+")"
        return retstr



    with open(path2file, "w") as file:
        for blobs in allblobs:
            for blob in blobs:
                line2file = lines2string((blob.getLines()))
                file.write(line2file)
            file.write("\n")


def file2Blobs(path2file):
    Blobs = []

    def lines2blobs(line):
        blobs = []
        regex = r"((\d+),?)+"
        #regex = r"((\d+),?){3}"
        matches = re.finditer(regex, line)
        for match in matches:
            l  = match.group().split(",")
            blob = Blob()
            cl = []
            for d in l:
                if len(cl) == 3:
                    blob.add(tuple(cl))
                    cl = []

                if d.isdigit():
                    cl.append(int(d))
            if len(cl) == 3:
                blob.add(tuple(cl))
            if len(blob) > 0:
                blobs.append(blob)
        return blobs



    with open(path2file) as file:
        line = file.readline()
        cnt = 1
        while line:
            line = file.readline()
            cnt += 1
            yield lines2blobs(line)
            



