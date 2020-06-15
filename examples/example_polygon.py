import os,sys,argparse
import numpy as np
import scipy.linalg


lib_path = os.path.abspath(os.path.join(__file__, '..', ".."))
sys.path.append(lib_path)

from libs.imProcess import *
from libs.MF import *
from libs.matProcess import *

import matplotlib.pyplot as plt
import matplotlib
from copy import copy

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations



def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("-s", "--save", help='Sauvegarde la matrice construite', type=str)

    args = parser.parse_args()
    args.process = True
    args.no_treat = True
    args.load = "examples/HST.npy"

    args.fantom = True

def main():

  global args
  
  if args.load:
    DATA = np.load(args.load)
  else:
    replace_special_characters(args.images_path)
    DATA = build_data_matrix(args.images_path,20)
    if args.save:
        np.save(args.save, DATA)

  if args.process:   
    print("DATA is real :", np.all(DATA == np.real(DATA)))
    data_reduced = reduction(DATA)
    valp, espp = process_matrix(DATA)
    sorted_valp = sort_eigenvalues(valp)
    print('shape DATA :', DATA.shape)
    print('shape data_reduced :', data_reduced.shape)
    print('shape valeurs propres :', valp.shape)
    print('shape vecteurs propres :', espp.shape)
    print('somme des vp :', np.sum(valp), "pourcentage des 2 premieres :", sorted_valp[0][1] + sorted_valp[1][1])
    # print('tableau des vp :', valp)
    #eigenvalues_histogram(valp, 5)
    new_DATA = compute_new_data_matrix(DATA, espp, valp, 5)
    DATA = new_DATA
    #polygon = [(-20,10),(10,10),(10,0),(-20,0)]
    polygon = [(-20,8),(0,12),(9,10),(10,0),(0,3)]
    size_window = [10,7]
    fig = plt.figure(figsize = (*size_window,))
    fig.add_subplot(221)
    for i, indiv in enumerate(DATA):
      x1 = indiv[0]
      y1 = indiv[1]
      plt.scatter(x1,y1, c='red')
    plt.grid()
    plt.title('Avant')
    #plt.xlabel(r"Projection sur $X'_1$ (en unité de $\sigma'_1$)")
    #plt.ylabel(r"Projection sur $X'_2$ (en unité de $\sigma'_2$)")
    fig.add_subplot(222)
    for i, indiv in enumerate(DATA):
      x1 = indiv[0]
      y1 = indiv[1]
      if polygon == None or is_in_polygon([x1],[y1],polygon):
        plt.scatter(x1,y1, c='red')
    plt.grid()
    plt.title('Uniquement l"interieur du polygon')
    #plt.xlabel(r"Projection sur $X'_1$ (en unité de $\sigma'_1$)")
    #plt.ylabel(r"Projection sur $X'_2$ (en unité de $\sigma'_2$)")
    print('shape new_DATA :', new_DATA.shape)
    fig.add_subplot(223)
    for i, indiv in enumerate(DATA):
      x1 = indiv[0]
      y1 = indiv[1]
      if polygon == None or is_in_polygon([x1],[y1],polygon):
        plt.scatter(x1,y1, c='blue')
      else:
        plt.scatter(x1,y1, c='red')

    plt.grid()
    plt.title('Interieur en bleu')
    #plt.xlabel(r"Projection sur $X'_1$ (en unité de $\sigma'_1$)")
    #plt.ylabel(r"Projection sur $X'_2$ (en unité de $\sigma'_2$)")
    print('shape new_DATA :', new_DATA.shape)
    ax = fig.add_subplot(224)
    for i, indiv in enumerate(DATA):
      x1 = indiv[0]
      y1 = indiv[1]
      if polygon == None or is_in_polygon([x1],[y1],polygon):
        plt.scatter(x1,y1, c='red')
      else:
        plt.scatter(x1,y1, c='red')
    poly = plt.Polygon(polygon, fill=False)
    ax.add_patch(poly)
    plt.grid()
    plt.title('yesyes')
    #plt.xlabel(r"Projection sur $X'_1$ (en unité de $\sigma'_1$)")
    #plt.ylabel(r"Projection sur $X'_2$ (en unité de $\sigma'_2$)")
    print('shape new_DATA :', new_DATA.shape)
    plt.show()


if __name__ == "__main__":
    init_args()
    main()
    
