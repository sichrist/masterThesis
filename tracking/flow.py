import cv2
import numpy as np
import os

def quiver(img,flow,step=5):
    
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    

    fy, fx = flow[y, x].T

    """ Normalize len = 1 """
    fy_n = []
    fx_n = []
    x_n = []
    y_n = []
    
    for i,(y_,x_) in enumerate(zip(fy,fx)):
        if y_ == x_ == 0:
            
            #fy_n.append(y_)
            #fx_n.append(x_)
            continue
        
        y1 = y_ / (np.abs(y_)+np.abs(x_))
        x1 = x_ / (np.abs(y_)+np.abs(x_))
        
        fy_n.append(y1)
        fx_n.append(x1)
        
        x_n.append(x[i])
        y_n.append(y[i])
    if len(img.shape) == 3:
        vis = img
    else:
        vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if len(x_n) == 0:
        return vis
        
    """                      """
    x_n  = np.array(x_n)
    y_n  = np.array(y_n)
    fx_n = np.array(fx_n)
    fy_n = np.array(fy_n)
    fx_n = fx_n * 10
    fy_n = fy_n * 10
    #lines = np.vstack([x, y, x + fx_n, y + fy_n]).T.reshape(-1, 2, 2)
    lines = np.vstack([x_n, y_n, x_n + fx_n, y_n + fy_n]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    
    cv2.polylines(vis, lines, 0, (255, 255, 255))
    
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


def draw_flow(imgs,flow, step=5):

            h, w = imgs.shape[:2]
            y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)

            lines = np.int32(lines + 0.5)
            #vis = img.copy()
            vis = np.zeros_like(imgs)

            #vis = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, lines, 0, (255, 255, 255),2)
            #for (x1, y1), (x2, y2) in lines:
            #    cv.circle(vis, (x1, y1), 1, (255, 255, 0), -1)
            return vis
