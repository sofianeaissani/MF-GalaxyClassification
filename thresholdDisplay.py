import sys
import matplotlib.pyplot as plt
from libs.imProcess import *
#from libs.MF import *
import argparse
import numpy as np
from scipy import ndimage, misc
import imageio


def main(myFile, cutout_size):

    global args

    a = get_image(myFile)[0]
    NUMAX = np.max(a)
    NUMIN = np.min(a)
    print("min :", NUMIN, "max :", NUMAX)

    """b = a.copy()
    c = a.copy()
    d = a.copy()
    e = a.copy()"""

    a_ = second_inflexion_point(a, cutout_size)
    a__ = second_inflexion_point(a_, cutout_size)

    plt.subplot(1, 2, 1)
    plt.imshow(a_)

    plt.subplot(1, 2, 2)
    plt.imshow(a__)

    plt.show()


"""
    a_ = a >= (NUMAX-NUMIN)/6
    b = b >= 2*(NUMAX-NUMIN)/6
    c = c >= 3*(NUMAX-NUMIN)/6
    d = d >= 4*(NUMAX-NUMIN)/6
    e = e >= 5*(NUMAX-NUMIN)/6



    ### Plotting

    size_window = [8, 5]

    fig = plt.figure(figsize = (*size_window,))

    fig.add_subplot(253)
    plt.imshow(a, cmap="rainbow")

    fig.add_subplot(256)
    plt.imshow(a_, cmap="binary")

    fig.add_subplot(257)
    plt.imshow(b, cmap="binary")

    fig.add_subplot(258)
    plt.imshow(c, cmap="binary")

    fig.add_subplot(259)
    plt.imshow(d, cmap="binary")

    fig.add_subplot(2,5,10)
    plt.imshow(e, cmap="binary")

    plt.show()"""


parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in format FITS or DAT', type=str)
    parser.add_argument("cutout_size", help='cutout size', type=int)
    args = parser.parse_args()

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.file, args.cutout_size)