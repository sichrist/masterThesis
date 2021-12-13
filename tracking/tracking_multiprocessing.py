import sys
import os
import cv2
from Tracking import VideoStreamer,NN_Tracker, resize, com, show_image,calcFlow,toGray


import numpy as np
import cupy as cp
from multiprocessing import Process, Queue, Lock, Semaphore, Value, Condition
from time import time
import cupyx.scipy.ndimage as cpyx
import scipy



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



def producer(queue,viewerQeueu,processes,path2Video,lock):
    
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
            img = mean_img - img
            img = toGray(img)
            img = cpyx.grey_closing(img,size=(5,5))
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
            
            
    except KeyboardInterrupt:
        del img
        del mean_img
        cp._default_memory_pool.free_all_blocks()
    del mean_img
    cp._default_memory_pool.free_all_blocks()

def viewer(blobqueue,bloblock,viewerQeueu,nProcesses,actual_frame,frame_condition,lock,blobfunction):
    
    frame_needed = 0
    from time import sleep
    print("Viewer")
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
    try:
        
        while True:


            with actual_frame.get_lock():
                actual_frame.value = frame_needed

            wake_up(frame_condition,lock)
            start = time()
            try:
                frame,offset,total,nbr,blob = blobqueue.get(block=False,timeout=0.1)
                #with lock:
                    #print("Viewer got frame: ",frame," slicenbr: ",nbr)
                    #sleep(0.01)
                #print()
            except :
                continue
            blobs.append(blob)
            slices += 1

            msg.append(frame)

            if slices == total:

                
                slices = 0

                img_frame,img = viewerQeueu.get()
 
                """
                def process_slices(blobs,img,msg):
                    blobs = [x for sublist in blobs for x in sublist]
                    blobs = sorted(blobs, key=lambda tup:tup[2])
                    for xs,xe,y in blobs:
                        img[y,xs:xe,:] = [255,0,0]
                    img = resize(img,scale_percent=40)
                    
                    #for i,m in enumerate(msg):
                    #    cv2.putText(img,str(m),(100,150+(i*25)), font, 1,(255,255,255),2,cv2.LINE_AA)

                    show_image(img)
                """
                blobfunction(blobs,img)

                msg = [frame_needed]
                
                blobs = []

                frame_needed += 1
                img_frame = None


    except KeyboardInterrupt:
        print("FINito")
