from tStudy.AI_Fish import *
import cv2
from time import sleep
from tStudy.Tracker import Linker
from tStudy.FlowMethod.extract import extract as extractFM
from tStudy.Image2Blobs import start, blobs2File, file2Blobs
from functools import partial
zebrafish_60 = "/mnt/HDD-1/schristo/CollectiveBehaviour/Data/60_zebrafish/group_1/video_04-07-17_10-07-50.000.avi"

rad = (2,4)
rad_from_center=250
img_shape = (1200,1200,3)
nbr = 60

def param4Circular():
    import math
    def init_function(img_shape,rad,nbr):
        
        h,w = img_shape[:2]
        middle = ((h//2),(w//2))

        x = np.random.randint(low=middle[0]-rad,high=middle[0]+rad,size=nbr)
        y = np.random.randint(low=middle[1]-rad,high=middle[1]+rad,size=nbr)
        return x,y



    def circular_movement(point,angle,mu = 0,sigma = 5):
        
        #angle = angle + np.random.normal(mu, sigma)
        angle = math.radians(angle)

        h, w = img_shape[:2]
        jiggle_x = np.random.normal(0,25)
        jiggle_y = np.random.normal(0,25)

        origin = (((w+jiggle_x)//2),((h+jiggle_y)//2))

        if np.random.uniform(0,10) >= 9:
            origin += np.random.normal(0, 50,size = 2)

        ox, oy = origin
        px, py = point

        c = 2*math.sin(angle/2) 
        c += np.random.uniform(0,c)
        r = np.sqrt((origin[0]-point[0])**2 + (origin[1]-point[1])**2) 
        p = c/(2*r)
        if p < -1:
            p = -1
        elif p > 1:
            p = 1

        new_angle = np.arcsin(p)*2

        angle = new_angle
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

        return qx.astype(np.int32), qy.astype(np.int32)


    return {
        "img_shape" : img_shape,
        "radius"    : rad,
        "start_area": (rad[1]*5,img_shape[0]-20),
        "velocity"  : 2,
        "direction" : [1,0],
        "nbr"       : nbr,
        "seed"      : None,
        "move_rout" : circular_movement,
        "angle"     : 160,
        "init_func" : partial(init_function,img_shape=img_shape,rad=rad_from_center,nbr=nbr),
        "frames"    : 3600
    }


def main():



    filename = "blobs.csv"
    from time import sleep
    Video_param = param4Circular()
    video = SchoolOfFishStream_AI(**Video_param,savename="Video0")
    #video = zebrafish_60
    
    
    allblobs  = start(video,threshold = 20,use_closing = True, use_sobel=False)

    for blobs in allblobs:
        img = np.zeros(img_shape)
        for blob in blobs:

            blob.blob2img(img)
        cv2.imshow("windowname",img)
        sleep(1/30)
        if cv2.waitKey(25) & 0XFF == ord('q'):
            raise KeyboardInterrupt
    blobs2File(allblobs,filename)
    
    for blobs in file2Blobs(filename):
        img = np.zeros(img_shape)
        for blob in blobs:
            blob.blob2img(img)
            blob.outline()
            print()
        cv2.imshow("windowname",img)
        sleep(1/30)
        if cv2.waitKey(25) & 0XFF == ord('q'):
            raise KeyboardInterrupt


if __name__ == '__main__':
    main()

