import numpy as np
from minkfncts2d import MF2D

def calcul_fonctionelles(file1, max_treshold, nb_points = 100):
    """Source: https://github.com/cefarix/minkfncts2d , https://iopscience.iop.org/article/10.1088/1742-5468/2008/12/P12015 """
    F = []
    U = []
    Chi = []
    for threshold in np.linspace(0.0, max_treshold, nb_points):
        (f, u, chi) = MF2D(file1, threshold)
        F.append(f)
        U.append(u)
        Chi.append(chi)
    return F, U, Chi

def func_col(arg):
    """ Définit la couleur des fonctionelles, pour que cela reste uniforme """
    if arg == "f":
        return [1,0,0]
    elif arg == "u":
        return [0,0.6,0]
    elif arg == "chi":
        return [0,0,1]
    else:
        return [0,0,0]


def condition_near_0(a,b,c):
    """ Determine quand la fonctionnelle est très petite """
    return a < 0.01 and b < 0.01 and c < 0.01



def get_start_end(f):
    """ Récupére le seuil auquel la fonctionnelle devient petite """
    s,e = 0,len(f)-1
    for i,v in enumerate(f):
        if i > 2:
            if condition_near_0(f[i-2],f[i-1],f[i]) and e == len(f)-1:
                e = i
    return s,e
    

def crop_functional(x,f):
    """ Tronque la fonctionnelle """
    s,e = get_start_end(f)
    x = x[s:e]
    f = f[s:e]
    return x,f

def normalize_on_x(x,f):
    s,e = x[0],x[-1]
    x,f = crop_functional(x,f)
    x = np.linspace(s, e, len(x))
    return x,f

def coef_normalization_functional(f):
    return np.max(np.abs(f))