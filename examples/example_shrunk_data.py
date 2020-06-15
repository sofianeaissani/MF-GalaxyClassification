import os,sys,argparse
import numpy as np
import scipy.linalg


lib_path = os.path.abspath(os.path.join(__file__, '..', ".."))
sys.path.append(lib_path)

from libs.imProcess import *
from libs.MF import *
from libs.matProcess import *
import libs.matProcess as mt

import matplotlib.pyplot as plt
import matplotlib
from copy import copy

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("-p", "--process", help='',action="store_true")
    parser.add_argument("-n", "--no_treat", help="Pas de traitement et renomage des fichiers (si vous l'avez deja fait)",action="store_true")
    parser.add_argument("-l", "--load", help='Ne calcule pas de matrice et load une matrice déjà faite', type=str)
    parser.add_argument("-s", "--save", help='Sauvegarde la matrice construite', type=str)

    args = parser.parse_args()
    args.no_treat = True
    args.load = "examples/HST.npy"
    args.process = True

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
    polygon = [(-20,8),(0,12),(9,10),(10,0),(0,3)]
    polygon2 = [(-5,3),(13,1),(13,-17),(-1.5,-7.7)]
    #plot_DATA_2D(new_DATA,polygon)
    print(new_DATA.shape)
    inxs,shrunk_NEW_DATA = get_in_polygon(new_DATA, polygon)
    shrunk_DATA = []
    for i,indiv in enumerate(DATA):
      if i in inxs:
        shrunk_DATA += [indiv]
    shrunk_DATA = np.float64(shrunk_DATA)

    inxs2,shrunk_NEW_DATA2 = get_in_polygon(new_DATA, polygon2)
    shrunk_DATA2 = []
    for i,indiv in enumerate(DATA):
      if i in inxs2:
        shrunk_DATA2 += [indiv]
    shrunk_DATA2 = np.float64(shrunk_DATA2)
    print(len(inxs))
    print('shape new_DATA :', new_DATA.shape)
    if False:
      size_window = [10,8]
      fig = plt.figure(figsize = (*size_window,))


      mt.global_curve2(DATA)
      plt.title("Global curve, no polygon")
      plt.show()

    size_window = [12,8]
    fig = plt.figure(figsize = (*size_window,))
    
    ax = fig.add_subplot(241)
    michelll(new_DATA, polygon, ax)
    plt.title("tellmewhatyouwant")
    fig.add_subplot(242)
    #print(shrunk_DATA.shape)
    #print(shrunk_DATA[:,:2].shape)
    mt.global_curve2(shrunk_DATA[:,:100])
    plt.title("F")
    fig.add_subplot(243)
    mt.global_curve2(shrunk_DATA[:,100:200])
    plt.title("U")
    fig.add_subplot(244)
    mt.global_curve2(shrunk_DATA[:,200:300])
    plt.title("Chi")
    ax = fig.add_subplot(245)
    michelll(new_DATA, polygon2, ax)
    plt.title('whatyoureallyreallywant')  
    fig.add_subplot(246)
    mt.global_curve2(shrunk_DATA2[:,:100])
    plt.title("F")
    fig.add_subplot(247)
    mt.global_curve2(shrunk_DATA2[:,100:200])
    plt.title("U")
    fig.add_subplot(248)
    mt.global_curve2(shrunk_DATA2[:,200:300])
    plt.title("Chi")
    plt.tight_layout()
    plt.show()

def michelll(DATA, polygon,ax):
  for i, indiv in enumerate(DATA):
      x1 = indiv[0]
      y1 = indiv[1]
      plt.scatter(x1,y1, c='red')
      if is_in_polygon([x1],[y1],polygon):
        plt.scatter(x1,y1, c='blue')

  poly = plt.Polygon(polygon, fill=False, lw=3)
  ax.add_patch(poly)
  plt.grid()
  #plt.xlabel(r"Projection sur $X'_1$ (en unité de $\sigma'_1$)")
  #plt.ylabel(r"Projection sur $X'_2$ (en unité de $\sigma'_2$)")

if __name__ == "__main__":
    init_args()
    main()
    
