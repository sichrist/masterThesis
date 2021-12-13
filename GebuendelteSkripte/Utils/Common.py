import itertools
import numpy as np
import copy
import cv2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    RED  = '\033[1;31m'
    UNDERLINE = '\033[4m'



def permut(a,b):
    perm = [ [*r] for r in itertools.product(a, b)]
    return perm

def rotate_degree(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def turnmax(v1,v2,theta):
    angle = angle_between(v1,v2)*180/np.pi
    if angle>theta:
        direction = CrossProduct2d(v1,v2)
        d = 1
        if direction < 0:
            d = -1
        return rotate_degree(v1,degrees=theta*d)
    return v2

def getAllPermutations(array):
    p = array[0]
    for i in range(1,len(array)):
        p = permut(p,array[i])
        for i,a in enumerate(p):
            if type(a[0]) == list:
                p[i]=[*a[0],a[1]]
    return p

def calcSpeed(data):
    speed = []
    for i in range(1,len(data)):
        v = np.sqrt(np.sum((data[i]-data[i-1])**2,axis=-1))
        speed.append(v)

    return np.array(speed)

def calcDistances(data):
    distances = []
    distances_min = []
    for i in range(len(data)):
        pos_atm = data[i]
        a = pos_atm.repeat(len(pos_atm),0).reshape(len(pos_atm),len(pos_atm),2)

        dist = np.sqrt(np.sum((a.transpose(1,0,2)-a)**2,axis=-1))

        if np.isnan(dist).any():
            dist = np.nan_to_num(dist) 

        distances.append(np.array(dist))
        dist[dist == 0] = np.inf

    return np.array(distances).astype(np.float)

def getDistances(path2file):
    path2dist = path2file.replace(path2file.split("/")[-1],"distances")
    try:
        speed = np.load(path2dist+".npz",allow_pickle=True)

    except:

        data = getRealData(path2file)
        print(path2file)
        tr= calcDistances(data)
        np.savez(path2dist,distance=np.array(tr,dtype=object))
    t = np.load(path2dist+".npz",allow_pickle=True)
    return t["distance"]

def getSpeed(path2file):
    path2speed = path2file.replace(path2file.split("/")[-1],"speed")
    try:
        speed = np.load(path2speed+".npz",allow_pickle=True)["speed"]

    except:
        data = getRealData(path2file)

        tr = np.array(calcSpeed(data))
        np.savez(path2speed,speed=tr)

    return np.load(path2speed+".npz",allow_pickle=True)["speed"]

def calcTurninRate(data):
    all_angles = []
    for i in range(2,len(data)):
        pos_prev_last = data[i-2]
        pos_last = data[i-1]
        pos_atm  = data[i]
        
        direction_a = pos_atm-pos_last
        direction_b = pos_last-pos_prev_last
        angles = [np.rad2deg(angle_between(d1,d2)) for d1,d2 in zip(direction_b,direction_a)]
        all_angles.append(angles)
    return all_angles


def getTurningrate(path2file):
    path2turningrate = path2file.replace(path2file.split("/")[-1],"turningrate")
        
    try:
        tr = np.load(path2turningrate+".npz",allow_pickle=True)["turningrate"]
    except:
        data = getRealData(path2file)
        tr = np.array(calcTurninRate(data))
        np.savez(path2turningrate,turningrate=tr)

    return np.load(path2turningrate+".npz",allow_pickle=True)["turningrate"]

def VecLen(V): 
    return np.sqrt(V[0]**2 + V[1]**2) 


def dist(p1,p2):
    """
        returns euklidean distance between points p1 and p2
    """
    return VecLen(np.array(p1)-np.array(p2))


def split(a, n):
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def normalize_Vec(V):

    l = VecLen(V)
    if np.sum(V) == 0:
        return np.array([0,0])
    return V / VecLen(V)


def lin(d,coeff):
    if d > coeff:
        return 0.0
    return float(d/coeff)


def invlin(d,coeff):
    if d > coeff:
        return 0.0
    return 1 - lin(d,coeff)


def e(x,coeff):
    if x > coeff:
        return 0
    a = np.exp((-(x/coeff)))
    return float(a)

 
def direction(center,p):
    return [center[0]-p[0],center[1]-p[1]]


def l2error(y,pred):
    diff = np.array(y) - np.array(pred)
    diff2 = diff**2
    return np.sum(diff2)/len(y)


def FR(x,R):
    if (x > R) or (x <= 0):
        return 0.0
    return x*(-1/R) + 1


def FO(x,R,O,A):
    if O == R:
        O+=1
        A+=1
    if A == O:
        A+=1

    if (x<R) or (x>A):
        return 0.0
    if x<= O:
        return (1/(O-R))*x-(R/(O-R))
    return (1/(O-A))*x-(A/(O-A))


def FA(x,O,A):
    if A == O:
        A+=1
    if x<O or x>2*A-O:
        return 0.0
    if x<A:
        return (1/(A-O))*(x-O)
    return(1/(A-O))*(-x+A)+1


def com(data):
    x = copy.copy(data)
    x = x[~np.isnan(x).any(axis=1)]
    return np.sum(x,axis=0)/len(x)


def CrossProduct2d(v1,v2):
    return v1[0]*v2[1] - v2[0]*v1[1]


def rotate(origin, point, angle):
    import math

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if np.isnan(qx):
        qx = 0
    if np.isnan(qy):
        qy = 0

    return np.array([np.int(qx), np.int(qy)])


def getRealData(path2Data):
    
    traj_data = np.load(path2Data,allow_pickle=True).item(0)
    trajectorie_data = traj_data["trajectories"]
    return trajectorie_data

def resize(img,scale_percent = 20):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img,dim)

def show_in_Notebook(a, fmt='jpeg',resizeIMG=True):
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
    if a.shape[0] > 1900 and resizeIMG == True:
        a = resize(a,scale_percent=30)
    PIL.Image.fromarray(a).save(faux_file, fmt)
    clear_output(wait=True)
    imgdata = Image(data=faux_file.getvalue())
    display(imgdata)

pixelToCm = 0.016
