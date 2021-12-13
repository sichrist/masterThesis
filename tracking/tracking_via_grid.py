import numpy as np 
import cv2



class Cell:
    def __init__(self,pos): 
        self.pos = pos
        self.Agents = []
        self.current    = -1
        
    def getPos(self):
        return self.pos
    
    
    def getNbr(self):
        return len(self.Agents)
    

    def getAgents(self):
        return self.Agents

    
    def append(self,agent):
        self.Agents.append(agent)

        
    def __eq__(self,other):
        pos = other.getPos()
        if self.pos[0] == pos[0] and self.pos[1] == pos[1]:
            return True
        return False
    
    def __len__(self):
        return len(self.Agents)
    
    def index(self):
        return self.pos[0],self.pos[1]
    
    def dist(self,other):
        p_x,p_y = self.pos
        x,y     = other.index()
        return np.sqrt((x - p_x)**2 + (y - p_y)**2)
     
    def remove(self,agent):
        self.Agents.remove(agent)
        
    def __getitem__(self, index):
        
        if index < len(self.Agents):
            return self.Agents[index]
        raise StopIteration
        
    
    

class GridStructure:
    def __init__(self,gridsize=5):
            self.gridsize = gridsize
            self.grid     = {}
            self.AgentsInGrid = []
            self.cells = []
            
            
    def Grid2img(self,img,thickness=1):
        y,x = img.shape[:2]
        
        for i in range(0,x+1,self.gridsize):
            cv2.line(img,(i,0),(i,x),(255,255,255),thickness)
            
            
        for i in range(0,y+1,self.gridsize):
            cv2.line(img,(0,i),(x,i),(255,255,255),thickness)
            
            
    def agents2Img(self,img,color_m = (255,0,0),color_r = (0,255,0),swap=False):
        for a in self.AgentsInGrid:
            if swap:
                y,x = a.index()
            else:
                x,y = a.index()

            cv2.circle(img, (x,y),2, color_m, -1)
            cv2.circle(img, (x,y),7, color_r, 2)
        return img

    
    def filled2Img(self,img,color=(255,0,0)):
        for a in self.AgentsInGrid:
            y,x = self.getCell(a)
            x = x*self.gridsize
            y = y*self.gridsize
            img[x:x+self.gridsize+1,y:y+self.gridsize+1,0] += color[0]
            img[x:x+self.gridsize+1,y:y+self.gridsize+1,1] += color[1]
            img[x:x+self.gridsize+1,y:y+self.gridsize+1,2] += color[2]
        return img


    def corrCell(self,cell):
        x,y = cell.index()
        if x not in self.grid:
            return None
        if y not in self.grid[x]:
            return None
        
        return self.grid[x][y] 


    def getCell(self,agent):
        x,y = agent.index()
        x   = x//self.gridsize
        y   = y//self.gridsize
        return x,y


    def getOccupiedCells(self):
        return self.cells

            
    def addAgent(self,agent):
        x,y = agent.index()
        x   = x//self.gridsize
        y   = y//self.gridsize
        
        if x not in self.grid:
            self.grid[x] = {}
        if y not in self.grid[x]:
            self.grid[x][y]=Cell((x,y))
            self.cells.append(self.grid[x][y])
            
        self.grid[x][y].append(agent)
        self.AgentsInGrid.append(agent)


    def removeAgent(self,Agent):
        self.AgentsInGrid.remove(Agent)
        x,y = self.getCell(Agent)
        if x in self.grid:
            if y in self.grid[x]:
                self.grid[x][y].remove(Agent)
        
        
        
    def addAgents(self,listOfAgents):
        for agent in listOfAgents:
            self.addAgent(agent)
    
    
    def isEmpty(self,agent):
        if type(agent) is tuple:
            x,y = agent
        else:
            x,y = self.getCell(agent)
        if x not in self.grid:
            return True
        if y not in self.grid[x]:
            return True
        return False

    
    def getInRange(self,agent,max_distance):
        neighbourscells = int(max_distance // self.gridsize)
        return self.getNeighbours(agent,neighbourscells)        
        
        
    def getNeighbours(self,agent,neighbourcells = 1):
        x,y = self.getCell(agent)

        start_x = x-neighbourcells if x-neighbourcells >= 0 else 0
        end_x   = x+neighbourcells+1 
        
        start_y = y-neighbourcells if y-neighbourcells >= 0 else 0
        end_y   = y+neighbourcells+1
        
        n_l     = []
        for i in range(start_x,end_x):
            for j in range(start_y,end_y):
                if self.isEmpty((i,j)):
                    continue
                if i == x and j == y:
                    ns = []
                    al = self.grid[i][j].getAgents()
                    for f in al:
                        if f == agent:
                            continue
                        ns.append(al)
                    n_l += al
                else:
                    n_l += self.grid[i][j].getAgents()
        return n_l


IDS = [i for i in range(30000)]
import copy


class AIFish:
    def __init__(self,pos,rad,id=None,move_routine=None,angle = None):
        self.pos = pos
        if id is None:
            self.id  = IDS.pop()
        else:
            self.id = id

        self.rad = rad
        self.color = (255,255,255)
        self.velcity_sigma = 0.5
        self.taken = False
        self.move_routine = move_routine
        self.angle = angle


    def getPos(self):
        return self.pos


    def getID(self):
        return self.id

    
    def insideBoundaries(self,h,w):
        if self.pos[0] >= 0 and h >self.pos[0] \
        and self.pos[1] >= 0 and w >self.pos[1]:
            return True
        return False


    def fish2img(self,img,dontFill=False,color=None):
        
        fill = -1
        rad = self.rad
        
        if color is None:
            color = self.color
            
        if dontFill:
            fill = 1
            rad  = 5
        x,y = img.shape[:2]
        if self.insideBoundaries(y,x):            
            img = cv2.circle(img, self.pos, rad, color, -1)
            img = cv2.circle(img, self.pos, rad, (0,255,0), 1)
            img[self.pos[1]-1:self.pos[1]+2,self.pos[0]-1:self.pos[0]+2,:] = [0,255,0]
        return img


    def isTaken(self):
        return self.taken
    
    def index(self):
        return self.pos[0],self.pos[1]


    def move(self,direction,velocity,mu=0,sigma=1):
        
        if self.move_routine is None:

            x = np.abs(np.random.normal(mu, sigma, 1))
            y = np.random.normal(mu, sigma, 1)
            new_pos = [self.pos[0],self.pos[1]]
            velocity = velocity + np.random.normal(0, self.velcity_sigma, 1)
            
            new_pos[0] += int((direction[0] + x)*velocity)
            new_pos[1] += int((direction[1] + y)*velocity)
            self.pos = tuple(new_pos)
        else:
            
            self.pos = self.move_routine(self.pos,self.angle)

    def moveViaFlow(self,flow,window=None):
        if window == None:
            window = self.rad
        x,y    = self.pos
        part   = flow[y-window:y+window+1,x-window:x+window+1]
        mean   = np.mean(part,axis=(0,1))
        mean   = np.nan_to_num(mean)
        self.pos = (int(mean[0] + self.pos[0]),int(mean[1] + self.pos[1]))
        
        
    def dist2fish(self,fish):
        x,y = self.pos
        p_x,p_y = fish.pos
        return np.sqrt((x - p_x)**2 + (y - p_y)**2)
        
        
        
class SchoolOfFishStream_AI:
    def __init__(self,img_shape  = (480,640,3),
                      radius     = (15,20),
                      start_area = (20,640),
                      velocity   = 0.5,
                      direction  = [1,0],
                      nbr        = 10,
                      seed       = None,
                      move_rout  = None,
                      angle      = None,
                      init_func  = None,
                      frames     = None):
        
        self.img_shape  = img_shape
        self.radius     = radius
        self.start_area = start_area
        self.velocity   = velocity
        self.direction  = direction
        self.seed       = seed
        self.plain_img  = None
        self.school     = []
        self.nbr        = nbr
        self.frames     = frames
        self.angle      = angle
        self.frame      = 0
        np.random.seed(self.seed)   
        self.move_routine = move_rout
        self.init_fish(init_function=init_func)
        
        # if at least one fish inside boundaries
        self.inImage   = False
        
        self.history   = {}

        
    def init_fish(self,init_function=None):
        x,y       = self.start_area
        rmin,rmax = self.radius
        if init_function is None:
            pos_y = np.random.uniform(low=self.radius[1], high=y-self.radius[0], size=self.nbr).astype(np.int)
            pos_x = np.random.uniform(low=self.radius[1], high=x-self.radius[0], size=self.nbr).astype(np.int)
        else:
            pos_y,pos_x = init_function()

        r = np.random.uniform(low=rmin, high=rmax, size=self.nbr).astype(np.int)
        self.school = [AIFish(pos,r[i],move_routine=self.move_routine,angle=self.angle) for i,pos in enumerate(zip(pos_x,pos_y))]
        

    def snapshot(self):
        
        for f in self.school:
            id_ = f.getID()

            if id_ not in self.history:
                self.history[id_] = []
                
            c = copy.copy(f)
            self.history[id_].append(c)
                
            
    def getHistory(self):
        return self.history
        
    def fish_inside_view(self):
        sum_ = 0
        
        for f in self.school:
            isin = f.insideBoundaries(self.img_shape[1],self.img_shape[0])
            sum_ += isin
        
        if sum_ > 0:
            return True
        return False


    def update(self):

        if self.plain_img is None:
            self.plain_img = np.zeros(self.img_shape)
            img = self.plain_img.copy()
            
            for f in self.school:
                f.fish2img(img)
            return img
        
        img = self.plain_img.copy()
        for f in self.school:
            f.move(self.direction,self.velocity)
            f.fish2img(img)

        return img
                    
        
    def __call__(self):

        if self.frames is not None:
            if self.frame >= self.frames:
                self.frame = 0
                return None

            self.frame += 1

        if not self.fish_inside_view():
            return None
                    
        img = self.update()
        self.snapshot()
        return img



def max_area2move(flow):
    """
        calculate the max distance a fish moves (according to its flow)
    """
    magnitude = np.sqrt((np.sum(flow**2,axis=-1))).flatten()
    idx = np.where(magnitude > 0)
    return magnitude[idx].max()

def mean_area2move(flow):
    """
        calculate average distance a fish moves
        add 2*std to this distance
    """
    magnitude = np.sqrt((np.sum(flow**2,axis=-1))).flatten()
    idx = np.where(magnitude > 0)
    return magnitude[idx].mean()+magnitude[idx].std()
    
"""
def assignFish(all_pred_cells,all_now_cells,grid_prev,flow,CORRECT,WRONG):

    assigned     = {}
    not_assigned = {}
    max_dist2move = max_area2move(flow)
    nothing_found = []
    for cell in all_now_cells:
        for fish in cell:
            fish_in_range = grid_prev.getInRange(fish,max_dist2move)
            isFish = None
            dist = max_dist2move
            for c in fish_in_range:
                c_copy = copy.copy(c)
                c_copy.moveViaFlow(flow)
                d = c_copy.dist2fish(fish)
                if d < dist:
                    dist = d
                    isFish = c
            if isFish is None:
                nothing_found.append(fish)
                continue
            grid_prev.removeAgent(isFish)
            assigned[isFish.getID()] = fish.getID()
    ctr = 0
    for key in assigned:
        if key == assigned[key]:
            CORRECT +=1
        else:
            WRONG += 1
        
        ctr+=1

    return assigned,CORRECT,WRONG
"""

def assignFish(all_pred_cells,all_now_cells,grid_prev,flow,CORRECT,WRONG):

    assigned     = {}
    not_assigned = {}
    max_dist2move = max_area2move(flow)
    nothing_found = []
    for cell in all_now_cells:
        for fish in cell:
            fish_in_range = grid_prev.getInRange(fish,max_dist2move)
            isFish = None
            dist = max_dist2move
            for c in fish_in_range:
                c_copy = copy.copy(c)
                c_copy.moveViaFlow(flow)
                d = c_copy.dist2fish(fish)
                if d < dist:
                    dist = d
                    isFish = c
            if isFish is None:
                nothing_found.append(fish)
                continue
            grid_prev.removeAgent(isFish)
            assigned[isFish.getID()] = (fish.getID(),fish)
    ctr = 0
    
    lonely_fish = [ a for a in grid_prev.AgentsInGrid]
    for f in lonely_fish:
        f.moveViaFlow(flow)
    grid_prev.AgentsInGrid = []
    
    def closest(f,l):
        d = 5000
        idx = 0
        fish = None
        for i in range(len(l)):
            dist = f.dist2fish(l[i])
            if dist < d:
                d = dist
                idx = i
                fish = l[i]

        if fish is None:
            fish = f
        else:
            l.remove(fish)
        return fish,l
            
    for fish in nothing_found:
            f,lonely_fish = closest(fish,lonely_fish)
            assigned[f.getID()] = (fish.getID(),fish)
            
    nothing_found = []
    for key in assigned:
        #print("{:2d} | {:3d} - {:3d}".format(ctr,key,assigned[key]))
        if key == assigned[key][0]:
            CORRECT +=1
        else:
            WRONG += 1
        
        ctr+=1

    #print()
        
    for f in nothing_found:
        WRONG +=1

    return assigned,CORRECT,WRONG