import numpy as np
import cv2
import sys
from fish import *
from flow import *
from tracking_via_grid import *
#from AI_fish import *
import numpy as np
from time import sleep

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

def getData2Ttrack(Video_param,showframes=False):
    from time import sleep
    video = SchoolOfFishStream_AI(**Video_param)
    nxt = video()
    image_history = [nxt]
    flow_history  = []
    draw_flow_hist = []



    while True:
        
        prvs = nxt
        nxt = video()

        if nxt is None:
            break
        if len(nxt.shape)==3:
            nxt  = cv2.cvtColor(nxt.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if len(prvs.shape)==3:
            prvs  = cv2.cvtColor(prvs.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        image_history.append(nxt)
        flow = calcFlow(prvs,nxt,windowsize=5,polyexp=5)
        flow_history.append(flow)
        img = draw_flow(prvs,flow,step=10)
        draw_flow_hist.append(img.copy())
        show = np.concatenate((img,prvs),axis=1)
        if showframes:
            show_image(show)
    return video.getHistory(),flow_history,draw_flow_hist,image_history


def getAssignedPerFrame(hist,flow_history,gridsize = 50):

    keys   = [k for k in hist]
    

    CORRECT = 0
    WRONG   = 0

    assigned_per_frames = []

    for frame in range(len(flow_history)):
        
        # Agents in previous state
        agents_prev = [copy.copy(hist[k][frame])for k in keys]
        
        
        # Agents in next State
        agents_now = [hist[k][frame+1] for k in keys] 

        
        # Agents moved by flow
        agents_pred = [copy.copy(hist[k][frame])for k in keys]
        for a in agents_pred:
            if a is None:
                continue
            a.moveViaFlow(flow_history[frame])
            
        grid_prev = GridStructure(gridsize)
        grid_now  = GridStructure(gridsize)
        grid_pred = GridStructure(gridsize)
        
        grid_prev.addAgents(agents_prev)
        grid_now.addAgents(agents_now)
        grid_pred.addAgents(agents_pred)
        
        
        all_pred_cells = grid_pred.getOccupiedCells()
        all_now_cells  = grid_now.getOccupiedCells()
        assigned_fish,CORRECT,WRONG = assignFish(all_pred_cells,all_now_cells,grid_prev,flow_history[frame],CORRECT,WRONG)
        assigned_per_frames.append(assigned_fish)
    
    
    return assigned_per_frames,CORRECT,WRONG