import numpy as np 
import matplotlib.pyplot as plt
from libs.pic_process import *
from scipy import signal
from copy import copy

import argparse
import os,sys

def main(myFile):

    global args

    # Importer l'image dans file1. L'image doit être "normalisée" entre 0 et 255 comme d'hab avec coolrange

    file1, name, ext = get_image(myFile)
    file1 = np.clip(np.rint(file1), 0, 255)
    
    # Tracer l'histogramme de l'image : nb de pixels pour chaque valeur d'intensité (lissé)
    
    NbOccurs = []
    threshold = [i for i in range(256)]
    for i in threshold:
        Occurs_i = np.count_nonzero(file1 == i)
        NbOccurs.append(Occurs_i)
    
    kernel = signal.gaussian(28, 7)
    NbOccursSmooth = np.convolve(NbOccurs, kernel, mode='same')
    accroiss = np.diff(NbOccursSmooth, append=[0])
    accroissSmooth = np.convolve(accroiss, kernel, mode='same')
    second = np.diff(accroissSmooth, append=[0])
    secondSmooth = np.convolve(second, kernel, mode='same')

    size_window = [8,5]

    fig = plt.figure(figsize = (*size_window,))
    fig.add_subplot(131)
    plt.title("Galaxie")
    plt.imshow(file1, cmap="viridis")
    plt.colorbar()

    fig.add_subplot(132)
    #plt.plot(threshold, NbOccurs, color="lightblue")
    plt.plot(threshold, NbOccursSmooth, color="blue")
    #plt.plot(threshold, accroiss, color="yellow")
    plt.plot(threshold, accroissSmooth, color="gold")
    #plt.plot(threshold, second, color="orange")
    plt.plot(threshold, secondSmooth, color="red")
    plt.title("Histogramme des occurrences")
    plt.grid()
    plt.xlabel(r"Seuil $\nu$")
    plt.ylabel(r"Nombre d'occurences")

    # Trouver une approximation grossière du deuxième point d'inflexion

    threshold = np.argmin(secondSmooth)
    while secondSmooth[threshold+1] < 0:
        threshold += 1

    print(threshold)

    file2 = file1.copy()
    file2[file2 < threshold] = 0

    fig.add_subplot(133)
    plt.imshow(file2, cmap="viridis")
    plt.title("Réduction du bruit")

    
    plt.tight_layout()
    plt.show()




parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in format FITS or DAT', type=str)
    parser.add_argument("-dat", "--dat", action="store_true", help="Force the DAT format processing")
    args = parser.parse_args()

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.file)