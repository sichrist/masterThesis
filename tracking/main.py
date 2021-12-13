#!/home/schristo/.anaconda3/envs/swarm/bin/python
import cv2
from videoOPS import *
from fish import *
import os
PATH   = "/home/schristo/.steamhdd/Backup2021/Dokumente/collective_behaviour/Data/TransitioningBehaviourInSchoolingFish"

# 30 Fish (golden shiners )
Video1 = "pcbi.1002915.s013.m4v"
# 70 Fish (golden shiners )
Video2 = "pcbi.1002915.s014.m4v"
# 150 Fish (golden shiners )
Video3 = "pcbi.1002915.s015.m4v"
# 300 Fish (golden shiners )
Video4 = "pcbi.1002915.s016.m4v"





def pipeline():
    video = Video1

    vidcap = cv2.VideoCapture(os.path.join(PATH,video))
    success,image = vidcap.read()
    count = 0

    viewer = ImgViewer(fps=25)

    while success:   
        success,image = vidcap.read()
        count += 1
        if count <= 130:
            continue 

        if success:
            #image = to_gray(image)
            image = cluster_fish(image)
            viewer(image)

        if cv2.waitKey(25) & 0XFF == ord('q'):
            break

    vidcap.release()




def main():
    pipeline()

if __name__ == '__main__':
    main()