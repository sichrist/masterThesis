import numpy as np


class Blob:
    def __init__(self,line=None):

        if line is not None:
            self.blobs = {line[-1]:[(line[0],line[1])]}
            self.lines = [line]

        else:
            self.blobs = {}
            self.lines = []

        self.center = [None,None]
                
    def __gt__(self,other):
        if len(other) > len(self):
            return True
        return False

    def centerOfMass(self):
        i = 0
        xm = 0
        ym = 0
        for x0,x1,y in self.lines:
            xm += (x0+x1)//2
            ym += y
            i  += 1

        return xm//i,ym//i


    def getLines(self):
        return self.lines


    def add(self,line):
        x0,x1,y = line
        if y not in self.blobs:
            self.blobs[y] = [(x0,x1)]
        else:
            self.blobs[y].append((x0,x1))
        self.lines.append( line )

    def outline(self):

        if len(self.lines)<3:
            return None

        min_x = self.lines[0][0]
        min_y = self.lines[0][-1]

        for x0,_,y in self.lines:
            if x0 < min_x:
                min_x = x0
            if y < min_y:
                min_y = y


        p = (0,0,-1)
        c = (0,0,-1)
        n = (0,0,-1)

        for lx0,lx1,ly in self.lines:
            lx0 -= min_x
            lx1 -= min_x
            ly  -= min_y

            """
                call finalize else if lx0−l′x1≥1∧lx0≥cx0lx⁢0-lx⁢1′≥1∧lx⁢0≥cx⁢0, 
                we either skipped a few pixels in n or l starts before c even had valid pixels. 
                This means that all pixels x between
            """
            if n[-1] != ly:
                pass
                

            
        
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
            return False

        if int(y) not in self.blobs:
            return False
        for x0,x1 in self.blobs[int(y)]:
            if x <= x1 and x >= x0:
                return True
        return False


    def estimateOrientation(self):
        pass


    def addCenter(self,x,y):
        self.center = [x,y]


    def index(self):
        if self.center[0] is None:
            return self.centerOfMass()
        return int(self.center[0]),int(self.center[1])


    def __len__(self):
        l = 0
        for x0,x1,y in self.lines:
            l += np.abs(x1-x0)+1
        return l


    def blob2img(self,img,color = [0,0,255]):
        for (x0,x1,y) in self.lines:
            if len(img.shape) == 3:
                img[y,x0:x1] = color
            else:
                img[y,x0:x1] = [255]