import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D
import argparse
import os,sys

#sys.path.insert(1, 'libs/')

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

from libs.imProcess import *
from libs.MF import *

def main(myFile):

    global args

    file1, name, ext = get_image(myFile)
    name = name[:20]

    # Noise reduction
    file1 = second_inflexion_point(file1)

    # Scale on the nu axis
    max_nu = 255
    if type(args.max) == int:
        max_nu = args.max

    # Smoothing the image before computing MF
    if args.smooth:
        file1 = smooth_file(file1, args.smooth)

    # MF computing
    F, U, Chi = calcul_fonctionelles(file1, max_nu)

    # Graphics
    size_window = [8,5]

    fig = plt.figure(figsize = (*size_window,))
    fig.add_subplot(121)
    plt.title("Galaxie - "+name)
    print(file1)
    plt.imshow(file1, cmap="viridis")
    plt.colorbar()

    fig.add_subplot(122)
    x = np.linspace(0.0, max_nu, 150)
    a,b,c = 1,1,1
    if args.normalize:
        a = coef_normalization_functional(F)
        b = coef_normalization_functional(U)
        c = coef_normalization_functional(Chi)
    plt.plot(x, np.array(F)/a, color = func_col("f"))
    plt.plot(x, np.array(U)/b, color = func_col("u"))
    plt.plot(x, np.array(Chi)/c, color = func_col("chi"))
    plt.title("Fonctions de Minkowski")
    plt.legend([r"Aire $F$", r"Périmètre $U$", r"Caractéristique d'Euler $\chi$"], bbox_to_anchor =(1,-0.2), loc = "upper right")
    plt.xlabel(r"Seuil $\nu$")
    plt.ylabel(r"Fonctionnelles   $\dfrac{V_\mu(\nu)}{\max(|V_\mu(\nu)|}$")
    plt.tight_layout()

    # Fin
    if args.save:
        print(name)
        plt.savefig(args.save + "/" +name +".png")
    else:
        plt.show()



parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in format FITS or DAT', type=str)
    parser.add_argument("-s", "--save", help="save at the specified path without showing", type=str)
    parser.add_argument("-m", dest="max", help="maximum of the nu axis", type = int)
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the curves")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth level of the image", default=0)
    args = parser.parse_args()

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.file)
