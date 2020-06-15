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

def median_of_treat(origfile, treat = gaussian_noise):
    initial = True
    for j in range(10):

        if treat == gaussian_noise:
            tfile = treat(origfile, 25)
        elif treat == pepper_and_salt:
            tfile = pepper_and_salt(origfile, .01)
        elif treat == adaptive_poisson_noise:
            tfile = adaptive_poisson_noise(origfile, 1)
        elif treat == uniform_poisson_noise:
            tfile = uniform_poisson_noise(origfile, 25)
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

def main(myFolder):

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
        

        
        
        gauss = median_of_treat(file1, gaussian_noise)
        pepper = median_of_treat(file1, pepper_and_salt)
        adapt = median_of_treat(file1, adaptive_poisson_noise)
        homog = median_of_treat(file1, uniform_poisson_noise)
            
        gauss_file = gaussian_noise(file1,25)
        pepper_file = pepper_and_salt(file1,.01)
        adapt_file = adaptive_poisson_noise(file1,1)
        homoh_file = uniform_poisson_noise(file1,25)


        # Visuels
        
        size_window = [8,5]

        fig = plt.figure(figsize = (*size_window,))
        fig.add_subplot(256)
        plt.title("orig")
        plt.imshow(file1, cmap="viridis")

        fig.add_subplot(257)
        plt.title("gaussian")
        plt.imshow(gauss_file, cmap="viridis")

        fig.add_subplot(258)
        plt.title("pepper")
        plt.imshow(pepper_file, cmap="viridis")

        fig.add_subplot(259)
        plt.title("adaptative_poisson")
        plt.imshow(adapt_file, cmap="viridis")

        fig.add_subplot(2,5,10)
        plt.title("poisson homogène")
        plt.imshow(homoh_file, cmap="viridis")


        cols = ["red", "green", "m", "orange"]

        fig.add_subplot(241)
        x = np.linspace(0.0, max_lin, 100)
        plt.plot(x, np.array(F))
        
        big_l = [gauss, pepper, adapt, homog]

        for j,v in enumerate(big_l):
            plt.plot(x, np.array(v[0][0:100]), color= cols[j])
            plt.gca().fill_between(x, [v[0][k]+v[1][k] for k in range(100)], [v[0][k]-v[1][k] for k in range(100)], color= cols[j],alpha=0.5)
        plt.title("F")
        plt.xlabel("Threshold")

        fig.add_subplot(242)
        x = np.linspace(0.0, max_lin, 100)
        plt.plot(x, np.array(U))
        for j,v in enumerate(big_l):
            plt.plot(x, np.array(v[0][100:200]), color= cols[j])
            plt.gca().fill_between(x, [v[0][k]+v[1][k] for k in range(100,200)], [v[0][k]-v[1][k] for k in range(100, 200)], color= cols[j],alpha=0.5)
        plt.title("U")
        plt.xlabel("Threshold")

        fig.add_subplot(243)
        x = np.linspace(0.0, max_lin, 100)
        plt.plot(x, np.array(Chi))
        for j,v in enumerate(big_l):
            plt.plot(x, np.array(v[0][200:300]), color= cols[j])
            plt.gca().fill_between(x, [v[0][k]+v[1][k] for k in range(200,300)], [v[0][k]-v[1][k] for k in range(200, 300)], color= cols[j],alpha=0.5)
        plt.legend(["Sans traitement", "Gaussian", "P&S", "Adaptative Poisson", "Poisson homogène"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel("Threshold")
        plt.title("Chi")
        


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
    main(args.folder)
