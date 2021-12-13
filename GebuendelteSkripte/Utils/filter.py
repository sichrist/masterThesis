import numpy as np
from .Common import getAllPermutations

def laplaceFilter(array):
    f = np.zeros([3]*len(array.shape))
    s = f.shape
    t = tuple([s[i]//2 for i in range(len(s))])
    
    f[t] = -2*(np.sum(t))
    for i in range(len(t)):
        t1 = [*t]
        t2 = [*t]
        t1[i]-=1
        t2[i]+=1
        f[tuple(t1)] = 1
        f[tuple(t2)] = 1
    return f

def filterIT(array,filt):
    shape = [i+2 for i in array.shape]
    newarr = np.zeros(shape)
    filtered = np.zeros(array.shape)
    idcs = getAllPermutations([np.arange(1,s-1) for s in newarr.shape])

    c = tuple([slice(1,-1) for s in newarr.shape])
    newarr[c] = array

    for i in idcs:
        
        slice_idx = tuple([slice(idx-1,idx+2) for idx in i])
        filtered[tuple(np.array(i)-1)] = np.sum(newarr[slice_idx]*filt)
    return filtered