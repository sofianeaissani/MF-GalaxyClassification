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
import matplotlib.image as mpimg

from libs.pic_process import *
from libs.minkos import *
from libs.matrices3 import *
from PIL import Image

def median_of_treat(origfile, treat= gaussian_noise, parameter=1):
    initial = True
    for j in range(10):
        print(j)

        if treat == gaussian_noise:
            tfile = treat(origfile, parameter)
        elif treat == pepper_and_salt:
            tfile = pepper_and_salt(origfile, parameter)
        elif treat == adaptive_poisson_noise:
            tfile = adaptive_poisson_noise(origfile, parameter)
        elif treat == uniform_poisson_noise:
            tfile = uniform_poisson_noise(origfile, parameter)
        else:
            pass
        Ft, Ut, Chit = calcul_fonctionelles(tfile, 255)

        Ft = np.array(Ft)
        Ut = np.array(Ut)
        Chit = np.array(Chit)

        Ft = normaliser(Ft)
        Ut = normaliser(Ut)
        Chit = normaliser(Chit)

        fuchi = np.hstack((Ft, Ut, Chit))
        if initial:
            mtx = fuchi
            initial = False
        else:
            mtx = np.vstack((mtx,fuchi))
    mtx = mtx.T

    ly= []
    sy = []
    for i in range(len(mtx)):
        ly+= [np.median(mtx[i])]
        sy+= [np.std(mtx[i])]
    

    return ly, sy

def main(myFolder, noise_type):

    global args

    files = os.listdir(myFolder)

    for i,myFile in enumerate(files):
        if i >= 1:
            break

        name = myFile

        file1 = Image.open(myFolder + "/" + myFile).convert("L")
        max_lin= 255
        file1 = np.float64(file1)

        # Calcul des fonctionnelles
        F, U, Chi = calcul_fonctionelles(file1, max_lin)

        F = normaliser(F)
        U = normaliser(U)
        Chi = normaliser(Chi)
        
        if noise_type == "g":
            curve1 = median_of_treat(file1, gaussian_noise, parameter=10)
            curve2 = median_of_treat(file1, gaussian_noise, parameter=20)
            curve3 = median_of_treat(file1, gaussian_noise, parameter=30)
            curve4 = median_of_treat(file1, gaussian_noise, parameter=40)
            noisy_file1 = gaussian_noise(file1,5)
            noisy_file2 = gaussian_noise(file1,10)
            noisy_file3 = gaussian_noise(file1,15)
            noisy_file4 = gaussian_noise(file1,20)
            name = "Gaussian"
        
        elif noise_type == "ps":
            curve1 = median_of_treat(file1, pepper_and_salt, parameter=0.01)
            curve2 = median_of_treat(file1, pepper_and_salt, parameter=0.03)
            curve3 = median_of_treat(file1, pepper_and_salt, parameter=0.05)
            curve4 = median_of_treat(file1, pepper_and_salt, parameter=0.07)
            noisy_file1 = pepper_and_salt(file1, 0.01)
            noisy_file2 = pepper_and_salt(file1, 0.03)
            noisy_file3 = pepper_and_salt(file1, 0.05)
            noisy_file4 = pepper_and_salt(file1, 0.07)
            name = "P&S"
        
        elif noise_type == "pih":
            curve1 = median_of_treat(file1, adaptive_poisson_noise, parameter=1)
            curve2 = median_of_treat(file1, adaptive_poisson_noise, parameter=2)
            curve3 = median_of_treat(file1, adaptive_poisson_noise, parameter=3)
            curve4 = median_of_treat(file1, adaptive_poisson_noise, parameter=4)
            noisy_file1 = adaptive_poisson_noise(file1, 1)
            noisy_file2 = adaptive_poisson_noise(file1, 2)
            noisy_file3 = adaptive_poisson_noise(file1, 3)
            noisy_file4 = adaptive_poisson_noise(file1, 4)
            name = "Poisson inhomogène"

        elif noise_type == "ph":
            curve1 = median_of_treat(file1, uniform_poisson_noise, parameter=10)
            curve2 = median_of_treat(file1, uniform_poisson_noise, parameter=20)
            curve3 = median_of_treat(file1, uniform_poisson_noise, parameter=30)
            curve4 = median_of_treat(file1, uniform_poisson_noise, parameter=40)
            noisy_file1 = uniform_poisson_noise(file1, 10)
            noisy_file2 = uniform_poisson_noise(file1, 20)
            noisy_file3 = uniform_poisson_noise(file1, 30)
            noisy_file4 = uniform_poisson_noise(file1, 40)
            name = "Poisson homogène"


        # Visuels
        
        size_window = [8,5]

        fig = plt.figure(figsize = (*size_window,))
        fig.add_subplot(256)
        plt.title("Original")
        plt.imshow(file1, cmap="viridis")

        fig.add_subplot(257)
        plt.title(name+ " 1")
        plt.imshow(noisy_file1, cmap="viridis")

        fig.add_subplot(258)
        plt.title(name+ " 2")
        plt.imshow(noisy_file2, cmap="viridis")

        fig.add_subplot(259)
        plt.title(name + " 3")
        plt.imshow(noisy_file3, cmap="viridis")

        fig.add_subplot(2,5,10)
        plt.title(name+ " 4")
        plt.imshow(noisy_file4, cmap="viridis")


        cols = ["gold", "orange", "darkorange", "red"]

        fig.add_subplot(241)
        x = np.linspace(0.0, max_lin, 100)
        plt.plot(x, np.array(F), color="black")
        
        big_l = [curve1, curve2, curve3, curve4]

        for j,v in enumerate(big_l):
            plt.plot(x, np.array(v[0][0:100]), color= cols[j])
            plt.gca().fill_between(x, [v[0][k]+v[1][k] for k in range(100)], [v[0][k]-v[1][k] for k in range(100)], color= cols[j],alpha=0.5)
        plt.title(r"Aire $F$")
        plt.xlabel(r"Seuil $\nu$")

        fig.add_subplot(242)
        x = np.linspace(0.0, max_lin, 100)
        plt.plot(x, np.array(U), color="black")
        for j,v in enumerate(big_l):
            plt.plot(x, np.array(v[0][100:200]), color= cols[j])
            plt.gca().fill_between(x, [v[0][k]+v[1][k] for k in range(100,200)], [v[0][k]-v[1][k] for k in range(100, 200)], color= cols[j],alpha=0.5)
        plt.title(r"Périmètre $U$")
        plt.xlabel(r"Seuil $\nu$")

        fig.add_subplot(243)
        x = np.linspace(0.0, max_lin, 100)
        plt.plot(x, np.array(Chi), color="black")
        for j,v in enumerate(big_l):
            plt.plot(x, np.array(v[0][200:300]), color= cols[j])
            plt.gca().fill_between(x, [v[0][k]+v[1][k] for k in range(200,300)], [v[0][k]-v[1][k] for k in range(200, 300)], color= cols[j],alpha=0.5)
        plt.legend(["Sans traitement", name + " 1", name + " 2", name + " 3", name + " 4"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel(r"Seuil $\nu$")
        plt.title(r"Caractéristique d'Euler $\chi$")
        


        #   plt.tight_layout()

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
    parser.add_argument("folder", help='file in format FITS or DAT', type=str)
    parser.add_argument("noise", help='type of noise', type=str, choices=["g", "ph", "pih", "ps"])
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = float, default=40)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int)
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the curves")
    parser.add_argument("-dat", "--dat", action="store_true", help="Force the DAT format processing")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth", default=1)
    args = parser.parse_args()

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.folder, args.noise)
