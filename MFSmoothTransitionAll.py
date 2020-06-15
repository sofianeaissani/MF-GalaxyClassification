import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D
import argparse
import os,sys

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

sys.path.insert(1, 'libs/')

from pic_process import *
from minkos import *



def get_right(F,U,chi,arg):
    """
        Retourne la fonctionelle correspondant à l'argument et sa couleur
    """
    x = 0
    if arg == "f":
        x = F
    elif arg == "u":
        x = U
    elif arg == "chi":
        x = chi
    return x, func_col(arg)

def get_right_graph(F,U,chi,arg):
    """
        Retourne le graphe correspondant à l'argument
    """
    x = 0
    if arg == "f":
        x = F
    elif arg == "u":
        x = U
    elif arg == "chi":
        x = chi
    return x

def main(myFile):
    global args

    file1, name, ext = get_image(myFile, args.dat)

    # Réhausser le contraste linéairement
    if args.contrastLinear:
        file1 = contrastLinear(file1, args.contrastLinear)
        
    
    # Determiner l'interval de definition de la fonction
    max_lin = args.max

    size_window = [10,8]

    fig = plt.figure(figsize = (*size_window,))
    fig.add_subplot(221)
    plt.title("Galaxy - " + name)
    plt.imshow(file1, cmap="viridis")



    f_graph = fig.add_subplot(222)
    u_graph = fig.add_subplot(223)
    chi_graph = fig.add_subplot(224)

    x = np.linspace(0.0, max_lin, 100)

    
    # Calcul sans lissage
    F, U, Chi = calcul_fonctionelles(file1, max_lin)
    le_liste = ["f","u","chi"]
    for i in le_liste:
        h,col = get_right(F,U,Chi, i)
        b = coef_normalization_functional(h)
        if args.nonorm:
            b = 1
        temp_graph = get_right_graph(f_graph, u_graph, chi_graph, i)
        temp_graph.set_title(i)
        temp_graph.set_xlabel("Threshold")
        temp_graph.plot(x, np.array(h)/b, color=col + [1] )
        temp_graph.legend([i], bbox_to_anchor =(1,-0.2), loc = "upper right")

    # Calculs avec lissage
    for i in range(1,args.smooth):
        a_values = np.linspace(0.01,1,args.smooth+1)
        temp_file1 = smooth_file(file1, i)
        F, U, Chi = calcul_fonctionelles(temp_file1, max_lin)
        for j in le_liste:
            h,col = get_right(F,U,Chi,j)
            b = coef_normalization_functional(h)
            if args.nonorm:
                b = 1
            temp_graph = get_right_graph(f_graph, u_graph, chi_graph, j)
            temp_graph.plot(x, np.array(h)/b, color=col +[a_values[-i-1]] )
    
    #plt.title("2D Minkowski Functionals - Change with convolution")
    #plt.legend(le_liste, bbox_to_anchor =(1,-0.2), loc = "upper right")
    plt.tight_layout()

    # Fin
    if args.save:
        if args.name:
            name = args.name
        print(name)
        plt.savefig(args.save + "/" +name +".png")
    else:
        plt.show()

parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in fits format', type=str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = int, default=40)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int, default=40)
    parser.add_argument("-dat", "--dat", action="store_true", help="file is in dat format")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth", default = 1)
    parser.add_argument("-n", "--name", type = str, help="name of file")
    parser.add_argument("-f", "--functional", type = str, help="name of functional to show",choices=['f', 'u', 'chi'], default = "chi")
    parser.add_argument("-nonorm", "--nonorm", action="store_true",help="No normalisation")
    args = parser.parse_args()

    args.drawall = True

if __name__ == "__main__":
    init_args()
    main(args.file)
