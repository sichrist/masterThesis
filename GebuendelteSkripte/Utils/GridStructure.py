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