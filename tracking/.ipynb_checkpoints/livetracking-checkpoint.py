 
import sys
from tracking_multiprocessing import *
import numpy as np
from time import sleep

zebrafish_60 = "/mnt/HDD-1/schristo/CollectiveBehaviour/Data/60_zebrafish/group_1/video_04-07-17_10-07-50.000.avi"
TRAJ = "/mnt/HDD-1/schristo/CollectiveBehaviour/Data/zebrafish_trajectories/60/1/trajectories_wo_gaps.npy"
zebrafish_60_e = 60
fish_expected = zebrafish_60_e

fish_video = zebrafish_60
npzfile = np.load(TRAJ,allow_pickle=True)
data = npzfile.item(0)
TRAJECTORIES =  data['trajectories']
TRAJ_INDEX = 0

windowname = 'OpenCvFrame'
#cv2.namedWindow(windowname)
#cv2.moveWindow(windowname,2600,40)


def start(blobfunction):

    QueueSize = 150
    processes = 20
    queue = Queue(maxsize=QueueSize)
    blobqueue = Queue(maxsize=5000)
    lock  = Lock()


    # lock for consuming blobs
    blobLock        = Semaphore()
    actual_frame    = Value('i',0)
    consumers       = []
    threshold       = 20

    frame_condition=Condition(Lock())

    viewerQeueu = Queue()

    # start viewer first
    view = Process(target=viewer, args=(blobqueue, blobLock,viewerQeueu, processes,actual_frame,frame_condition,lock,blobfunction))
    #view.daemon = True
    view.start()


    for i in range(processes):
        p = Process(target=consumer, args=(queue, blobqueue,blobLock,lock,actual_frame,threshold,frame_condition))
        p.daemon = True
        consumers.append(p)


    prod = Process(target=producer, args=(queue, viewerQeueu,processes,fish_video,lock))
    prod.daemon = True
    prod.start()



    for c in consumers:
        c.start()

    try:
        view.join()

    except KeyboardInterrupt:
        for c in consumers:
            c.terminate()
            prod.terminate()
            view.terminate()
            cv2.destroyAllWindows()


class Blob:
    def __init__(self,line):
        self.blobs = {line[-1]:[(line[0],line[1])]}
        self.lines = [line]
        self.center = [None,None]
                
            
    def add(self,line):
        x0,x1,y = line
        if y not in self.blobs:
            self.blobs[y] = [(x0,x1)]
        else:
            self.blobs[y].append((x0,x1))
        self.lines.append( line )
        
    def partOfMe(self,line):
        x0,x1,y = line 
        if y not in self.blobs:
            return False

        for X0,X1 in self.blobs[y]:
            if x0 == X0 and x1==X1:
                return True
        return False
        
        
    def merge(self,blob):
        for l in blob.lines:
            self.add(l)
        
    def insideBlob(self,x,y):
        if y != y or x != x:
            print(y,x)
            return False

        if int(y) not in self.blobs:
            return False
        for x0,x1 in self.blobs[int(y)]:
            if x <= x1 and x >= x0:
                return True
        return False

    def addCenter(self,x,y):
        self.center = [x,y]

    def index(self):
        return int(self.center[0]),int(self.center[1])


    def __len__(self):
        l = 0
        for x0,x1,y in self.lines:
            l += np.abs(x1-x0)+1
        return l
            
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
    global TRAJ_INDEX
    global BLOB_HISTORY

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
                    print("NONE")
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

    SWARM_COM = [0,0]
    for x,y in TRAJECTORIES[TRAJ_INDEX]:
        img = cv2.circle(img, (int(x),int(y)), 20, (0,0,0), 2)
        SWARM_COM[0] += x
        SWARM_COM[1] += y

    SWARM_COM[0] /= len(TRAJECTORIES[TRAJ_INDEX])
    SWARM_COM[1] /= len(TRAJECTORIES[TRAJ_INDEX])
    img = cv2.circle(img, (int(SWARM_COM[0]),int(SWARM_COM[1])), 30, (128,0,0), -2)


    for i,blob in enumerate(blobs):
        BLOB_HISTORY.addBlob(blob,TRAJECTORIES[TRAJ_INDEX])
        if len(blob) < 200:
            continue
        nbr_blobs +=1

        for x0,x1,y in blob.lines:
            img[y,x0:x1,:] = [255,0,0]

        if len(blob) > ma:
            ma = len(blob)
            idx = i
      

    #for x0,x1,y in blobs[idx].lines:
    #    img[y,x0:x1,:] = [0,255,0] 
                
                       
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(ma), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(nbr_blobs), (20,550), font, 3, (0, 255, 0), 2, cv2.LINE_AA)


    def draw_traj(img):
        last_5 = BLOB_HISTORY(last=20)
        for key,blobhist in last_5:
            for i in range(len(blobhist)-1):
                start_point = blobhist[i][0]
                end_point = blobhist[i+1][0]
                img = cv2.line(img, (int(start_point[0]),int(start_point[1])), (int(end_point[0]),int(end_point[1])), (0,255,0), 2)

        return img


    def getNeighbours(img,search_distance=300):
        from tracking_via_grid import GridStructure

        grid = GridStructure()

        lastBlobs = BLOB_HISTORY(last = 1)
        for _,values in lastBlobs:
            blob = values[0][1]
            grid.addAgent(blob)


        for _,values in lastBlobs:
            blob = values[0][1]
            #img = cv2.circle(img, blob.index(), search_distance, (0,0,255), 1)

            in_range = grid.getInRange(blob,max_distance=search_distance)

            for n in in_range:
                img = cv2.line(img, blob.index(), (n.index()), (0,0,255), 2)



    img = draw_traj(img)
    getNeighbours(img)
    TRAJ_INDEX += 1
    img = resize(img,scale_percent = 30)
    cv2.imshow(windowname,img)
    if cv2.waitKey(25) & 0XFF == ord('q'):
        raise KeyboardInterrupt
        exit(-1)

    if cv2.waitKey(25) & 0XFF == ord(" "):
        print("Stop")
        sleep(5)
        while cv2.waitKey(25) & 0XFF != ord("m"): 
            continue
        print("Continue")




start(slices2blobs)

