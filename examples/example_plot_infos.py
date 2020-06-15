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
    args.process = True
    args.load = "npy/supertoi.npy"

    args.fantom = True

def main():

  global args
  
  if args.load:
    DATA = np.load(args.load)
    names = np.load(args.load+"cool.npy")
    print(DATA.shape)
    print(names.shape)
  else:
    replace_special_characters(args.images_path)
    DATA,names = build_data_matrix2(args.images_path,500)
    if args.save:
        np.save(args.save, DATA)
        np.save(args.save+"cool", names)

  if args.process:   
    all_the_data = extract_galaxies_data("csv_complet")
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

    full_plot_infos(new_DATA,all_the_data,names, polygon)

def full_plot_infos(new_DATA, physical_data,names, polygon = ""):

  all_the_data = physical_data
  #plot_DATA_2D(new_DATA,polygon)
  size_window = [10,8]
  fig = plt.figure(figsize = (*size_window,))
  fig.add_subplot(121)

  plot_cool_poly(new_DATA, polygon)
  fig.add_subplot(122)

  plot_infos(new_DATA, physical_data, names, polygon)

  plt.show()

def plot_infos(new_DATA, physical_data,names, polygon = ""):
  all_the_data = physical_data
  inxs, shrunk_data = get_in_polygon(new_DATA, polygon)
  print('shape new_DATA :', new_DATA.shape)

  #print()
  #sys.exit()
  michel = []
  for i in inxs:
    temp = names[i].replace("p",".")
    temp = key_name(temp)
    michel+=[all_the_data[temp]]
  #print(michel)
  list_key, physical_data = as_numpy(michel)
  out = treat_things(list_key, physical_data)
  
  txt = ["",""]
  i = 0
  full = math.ceil(len(out.keys())/2)
  for k,v in out.items():
    k = k.replace("_","\_")
    txt[i//full] += "$\\bf{"+str(k)+"}$" + ": \n"
    txt[i//full] += " "+ "MOY="+str(v["moy"])+"\n"
    txt[i//full] += " "+ "STD="+str(v["std"])+"\n"
    txt[i//full] += " "+ "MED="+str(v["med"])+"\n"
    i +=1
  plt.text(0,0,txt[0], fontsize=9)
  plt.text(0.6,0,txt[1], fontsize=9)
  plt.axis("off")

def plot_cool(data, list_keys, physical_data, key):
  size_window = [10,8]
  fig = plt.figure(figsize = (*size_window,))
  assert key in list_keys
  inx = list_keys.index(key)
  maxx = max(physical_data[inx])
  minn = min(physical_data[inx])
  alpha, beta, ceta = [],[],[]
  for i,indiv in enumerate(data): 
      x1 = indiv[0]
      y1 = indiv[1]
      value = physical_data[inx][i]
      color = value #(value - minn) / maxx
      alpha += [x1]
      beta += [y1]
      ceta += [color]
  plt.scatter(alpha, beta, c=ceta, cmap="viridis")
  plt.title("Repartition de " + key)
  plt.grid()
  plt.colorbar()
  #plt.gray()
  plt.show()

if __name__ == "__main__":
    init_args()
    main()
    
