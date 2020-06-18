import numpy as np
import scipy.linalg
import os,sys
from libs.MF import *
from libs.imProcess import *
from libs.phyProcess import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
from sklearn.cluster import KMeans
from random import shuffle


from matplotlib import colors
import matplotlib.pylab as plb
import matplotlib as mpl


def replace_special_characters(path):
  """ Remplace les caractères -, \\x1d, + par d'autres caractères dans les noms des fichiers de path """
  for i in os.listdir(path):
    if i[-3:] in ["txt","dat"]:
      n_name = i[:-4].replace("-", "_").replace("\x1d", "").replace("+", "_").replace(".","p")
      ext = i[-3:]
    elif i[-4:] == "fits":
      n_name = i[:-5].replace("-", "_").replace("\x1d", "").replace("+", "_").replace(".","p")
      ext = i[-4:]
    else:
      n_name = i
    os.rename(path+'/' + i,path+'/'+ n_name+'.'+ext)

def build_data_matrix(images_path, max_iter=300):
  """ Construit la matrice de données DATA contenant toutes les observations (MF sans CAS) de tous les individus
  - Entrée : chemin relatif vers le dossier contenant toutes les images
  - Sortie : matrice DATA au format n*p avec n le nombre d'individus et p le nombre de variables """ 

  initial = True

  images_list = os.listdir(images_path)
  
  for i,v in enumerate(images_list):
    print('index :', i)
    print('name :', v)
    ext = v[-4:]

    if i > max_iter:
      break
    if ext=="fits" or ext==".dat":
      print("Fichier trouvé, en cours de process")  
      image_file = images_path + "/" + v

      data_fonctionnelles = get_image(image_file)[0]
      data_fonctionnelles = second_inflexion_point(data_fonctionnelles)

      #data_fonctionnelles = cool_range(data_fonctionnelles)
      #data_fonctionnelles = smooth_file(data_fonctionnelles, 2)
      #data_fonctionnelles = contrastLinear(data_fonctionnelles[0], 10**4)

      F,U,Chi = calcul_fonctionelles(data_fonctionnelles, 256)
      F,U,Chi = np.array(F), np.array(U), np.array(Chi)
      
      F = normaliser(F)
      U = normaliser(U)
      Chi = normaliser(Chi)
      
      N = np.hstack((F,U,Chi))

      if initial:
        DATA = N
        initial = False
      else:
        DATA = np.vstack((DATA, N))

  return DATA

def normaliser2(func):
  a = np.min(func)
  func = func - a
  b = np.max(func)
  if b != 0:
    return func/b
  else:
    return func

def normaliser(func):
  a = np.max(np.abs(func))
  if a != 0:
    return func/a
  else:
    return func

def reduction(m):
  """Réduit la matrice m, c'est-à-dire :
  - Soustrait à chaque colonne sa moyenne
  - Divise chaque colonne par son écart-type.
  Puis renvoie une nouvelle matrice m' du même format que m."""

  matrix = np.copy(m.T)

  for i in range(matrix.shape[0]):
    matrix[i] = matrix[i] - np.mean(matrix[i])
  
  for i in range(matrix.shape[0]):
    std = np.std(matrix[i])
    if std != 0:
      matrix[i] = matrix[i]/std
    # else:
      # print("std = 0")
      
  return matrix.T

def process_matrix(DATA):
  """ Calcule les valeurs et vecteurs propres correspondant aux composantes principales de DATA.
  - Entrée : matrice initiale DATA au format n*p avec n le nombre d'individus et p le nombre de variables
  - Sortie : 2-tuple contenant la liste des valeurs propres et la liste des vecteurs propres
  Implémentation de la méthode tirée de 'Probabilités, statistiques et analyses multicritères', de Mathieu Rouaud.  """

  data_reduced = reduction(DATA)
  matrice_correlation = 1/data_reduced.shape[0] * np.dot(data_reduced.T, data_reduced)
  print("matrice_correlation is symetric :", np.all(matrice_correlation == matrice_correlation.T))
  print("matrice_correlation is real :", np.all(matrice_correlation == np.real(matrice_correlation)))
  val_et_espaces = scipy.linalg.eigh(matrice_correlation)

  return val_et_espaces     # pas forcément dans l'ordre souhaité

def sort_eigenvalues(valeursPropres):
  """ Rend les valeurs propres triées par ordre décroissant. 
  - Entrée : tableau des valeurs propres (format numpy)
  - Sortie : tableau contenant les 3-tuples (val, pourcentage, indice de l'espace propre correspondant)"""
  
  valeursPropres = valeursPropres.real
  for i in range(len(valeursPropres)):
    if np.isclose(0, valeursPropres[i]):
      valeursPropres[i] = 0
    assert valeursPropres[i] >= 0
  p = sum(valeursPropres)
  supertuples = [(valeursPropres[i], valeursPropres[i]/p,i) for i in range(len(valeursPropres))]
  supertuples.sort(reverse=True)

  return supertuples

def eigenvalues_plot(valeursPropres, n):
  """ Affiche la fonction de cumul des variances des n valeurs propres les plus importantes de valeursPropres 
  - Entrée : tableau des valeurs propres (format numpy)""" 
  assert n <= len(valeursPropres)

  valeursPropres = sort_eigenvalues(valeursPropres)

  fig = plt.figure()
  ax = fig.add_axes([0.1,0.1,0.8,0.8])
  valeursPropres = valeursPropres[:n] # Tronquer
  x,y = [],[]
  total = 0
  for j in range(len(valeursPropres)):
    val = valeursPropres[j]
    total += val[1]
    x += [j+1]
    y += [total]
  ax.scatter(x,y, marker=".", color="black")
  ax.plot(x,y, color="black")
  for j in range(len(valeursPropres)):
    plt.annotate(str(round(valeursPropres[j][1], 4)*100)+"%", xy=(j+0.5, y[j]), xytext=(j+0.5, y[j]+0.02), size=6)
  ax.bar(x,y, color="moccasin")
  plt.xlabel(r"Nombre de composantes principales $i$")
  plt.ylabel(r"Fraction de variance expliquée $\Sigma\delta_i/tr(\Delta)$")
  plt.show()

def compute_new_data_matrix(DATA, espp, valeursPropres, n):
  """ Calcule la nouvelle matrice de données évaluant chaque individu selon les nouvelles variables.\n
  Si n < len(valeursPropres), la nouvelle matrice comporte uniquement les n variables les plus dispersives.
  - Entrée : matrice de données initiale, tableau des espaces propres, tableau des valeurs propres (format numpy), nombre de variables voulues
  - Sortie : nouvelle matrice de données, éventuellement projetée sur un nombre restreint de variables""" 

  assert n <= len(valeursPropres)

  valeursPropres = sort_eigenvalues(valeursPropres)
  valeursPropres = valeursPropres[:n]
  
  new_DATA = np.dot(reduction(DATA),espp)   
  indexes = []
  for v in valeursPropres:
    indexes.append(v[2])
  new_DATA = new_DATA[:, indexes]

  return new_DATA

def plot_DATA_2D(DATA,inside_pol = None):
  """ Affiche la projection des individus dans le plan des 2 variables d'inertie maximale. """
  size_window = [5,5]
  fig = plt.figure(figsize = (*size_window,))
  for i, indiv in enumerate(DATA):
    x1 = indiv[0]
    y1 = indiv[1]
    if inside_pol == None or is_in_polygon([x1],[y1],inside_pol):
      plt.scatter(x1,y1, c='red')
  plt.grid()
  plt.title('Projections de chaque individu sur les 2\n premières composantes principales')
  plt.xlabel(r"Projection sur $X'_1/\sigma'_1$")
  plt.ylabel(r"Projection sur $X'_2/\sigma'_2$")

  plt.show()

def plot_cool_poly(DATA, polygon):
  for i, indiv in enumerate(DATA):
    x1 = indiv[0]
    y1 = indiv[1]
    if polygon == None or is_in_polygon([x1],[y1],polygon):
      plt.scatter(x1,y1, c='blue')
    else:
      plt.scatter(x1,y1, c='red')
  plt.grid()

def get_in_polygon(DATA, polygon):
  """ Récupère les indices et les éléments de DATA qui sont dans le polygone """
  out_inx = []
  shrunk_DATA = []
  for i, indiv in enumerate(DATA):
    x1 = indiv[0]
    y1 = indiv[1]
    if is_in_polygon([x1],[y1],polygon):
      out_inx += [i]
      shrunk_DATA += [indiv]
  return out_inx, np.float64(shrunk_DATA)

def plot_DATA_3D(DATA):
  """ Affiche la projection des individus dans l'espace des 3 variables d'inertie maximale. """
  size_window = [5, 5]
  fig = plt.figure(figsize = (*size_window,))
  ax = fig.add_subplot(111, projection='3d')
  for i, indiv in enumerate(DATA):
    x1 = indiv[0]
    y1 = indiv[1]
    z1 = indiv[2]
    ax.scatter(x1,y1,z1, c='red')
  ax.set_xlabel(r"Projection sur $X'_1/\sigma'_1$")
  ax.set_ylabel(r"Projection sur $X'_2/\sigma'_2$")
  ax.set_zlabel(r"Projection sur $X'_3/\sigma'_3$")
  # ax.set_title('Projections de chaque individu sur les 3\n premières composantes principales')

  plt.show()

def plot_DATA_3D_in_2D(DATA):
  """ Affiche la projection des individus dans l'espace des 3 variables d'inertie maximale. """
  size_window = [5, 5]
  fig = plt.figure(figsize = (*size_window,))
  ax = fig.add_subplot(111)
  l_x = []
  l_y = []
  l_c = []
  for i, indiv in enumerate(DATA):
    x1 = indiv[0]
    y1 = indiv[1]
    z1 = indiv[2]
    l_x.append(x1)
    l_y.append(y1)
    l_c.append(z1)
  q1 = np.quantile(l_c, 0.05)
  q3 = np.quantile(l_c, 0.95)
  n = 10
  part =(q3-q1)/n
  for i,v in enumerate(l_c):
    l_c[i] = clamp(q1,v,q3)

  # tell imshow about color map so that only set colors are used
  #ax.scatter(l_x,l_y,c=l_c)


  #catter_seq(fig,ax,l_x,l_y,l_c,q1,q3,10)
  scatter_cont(fig, ax, l_x, l_y, l_c)
  ax.set_xlabel(r"Projection sur $X'_1/\sigma'_1$")
  ax.set_ylabel(r"Projection sur $X'_2/\sigma'_2$")
  #ax.set_zlabel(r"Projection sur $X'_3$ (en unité de $\sigma'_3$)")
  # ax.set_title('Projections de chaque individu sur les 3\n premières composantes principales')

  plt.show()

def scatter_cont(fig,ax,x,y,c, mark="."):
  plt.scatter(x, y, c=c,cmap="viridis", marker=mark)
  plt.colorbar()

def scatter_seq(fig,ax, x,y,c,mi,ma,n):
  cmap = plt.cm.jet  # define the colormap
  # extract all colors from the .jet map
  cmaplist = [cmap(i) for i in range(cmap.N)]
  # force the first color entry to be grey
  #cmaplist[0] = (.5, .5, .5, 1.0)

  # create the new map
  cmap = colors.LinearSegmentedColormap.from_list(
      'Custom cmap', cmaplist, cmap.N)

  # define the bins and normalize
  bounds = np.linspace(mi,ma, n)
  norm = colors.BoundaryNorm(bounds, cmap.N)

  # make the scatter
  scat = ax.scatter(x, y, c=c,cmap=cmap, norm=norm)

  # create a second axes for the colorbar
  ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
  cb = mpl.colorbar.ColorbarBase (ax2, cmap=cmap, norm=norm,
  spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

def clamp(a,b,c):
  return min(max(a,b),c)

def is_in_polygon(x,y,pol):
  """ Vérifie si les points xy sont dans le polygone pol """
  points = np.vstack((x,y)).T
  p = Path(pol)
  grid = p.contains_points(points)
  return grid

def build_data_matrix2(images_path, max_iter=1000):
  """ Construit la matrice de données DATA contenant toutes les observations (MF sans CAS) de tous les individus
  - Entrée : chemin relatif vers le dossier contenant toutes les images
  - Sortie : matrice DATA au format n*p avec n le nombre d'individus et p le nombre de variables """ 

  initial = True

  images_list = os.listdir(images_path)
  list_of_names = []
  
  for i,v in enumerate(images_list):
    name = v.split(".")[0]
    ext = v[-4:]
    print('index :', i)
    print('name :', name)

    if i > max_iter:
      break
    if ext=="fits" or ext==".dat":
      list_of_names += [name]
      print("Fichier trouvé, en cours de process")
      image_file = images_path + "/" + v

      data_fonctionnelles = get_image(image_file)[0]
      data_fonctionnelles = second_inflexion_point(data_fonctionnelles)
      #data_fonctionnelles = smooth_file(data_fonctionnelles, 2)

      F,U,Chi = calcul_fonctionelles(data_fonctionnelles, 256)
      F,U,Chi = np.array(F), np.array(U), np.array(Chi)

      F = normaliser(F)
      U = normaliser(U)
      Chi = normaliser(Chi)

      N = np.hstack((F,U,Chi))

      if initial:
        DATA = N
        initial = False
      else:
        DATA = np.vstack((DATA, N))

  return DATA,list_of_names

def treat_things(list_keys, physical_data):
  out = {}
  for i,k in enumerate(list_keys):
    elmt = physical_data[i]
    out[k] = {"std":elmt.std(), "moy":elmt.mean(), "med":np.median(elmt)}
  return out

def as_numpy(physical_data):
  out = []
  list_keys = list(physical_data[0].keys())

  for elmt in physical_data:
    temp = []
    for k in list_keys:
      temp += [elmt[k]]
    out += [temp]
  
  out = np.float64(out)

  return list_keys, out.T

def find_clusters(new_DATA, nb_clusters):
  kmeans = KMeans(n_clusters=nb_clusters, n_init=50).fit(new_DATA)
  labels = kmeans.labels_
  inertia = kmeans.inertia_
  return labels, inertia

def plot_DATA_2D_with_clustering(DATA, nb_clusters, verbose=None):
  """A DETRUIRE"""
  """ Affiche la projection des individus dans l'espace des 3 variables d'inertie maximale avec clustering. """
  if nb_clusters > 26:
    print("Not enough markers to distinguish all the clusters.")
  labels, inertia = find_clusters(DATA, nb_clusters)
  print("Inertia :", inertia)
  K = np.max(labels)
  markerslist = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", r"$\mathcal{D}$", 
  r"$\mathcal{E}$", r"$\mathcal{F}$", r"$\mathcal{G}$", r"$\mathcal{H}$", r"$\mathcal{I}$", 
  r"$\mathcal{J}$", r"$\mathcal{K}$", r"$\mathcal{L}$", r"$\mathcal{M}$", r"$\mathcal{N}$",
  r"$\mathcal{O}$", r"$\mathcal{P}$", r"$\mathcal{Q}$", r"$\mathcal{R}$", r"$\mathcal{S}$",
  r"$\mathcal{T}$", r"$\mathcal{U}$", r"$\mathcal{V}$", r"$\mathcal{W}$", r"$\mathcal{X}$",
  r"$\mathcal{Y}$", r"$\mathcal{Z}$"]
  size_window = [5, 5]
  fig = plt.figure(figsize = (*size_window,))
  ax = fig.add_subplot(111)

  for k in range(K+1):  
    l_x = []
    l_y = []
    for i, label in enumerate(labels):
      if label==k:
        indiv = DATA[i]
        x1 = indiv[0]
        y1 = indiv[1]
        l_x.append(x1)
        l_y.append(y1)

    plt.scatter(l_x, l_y,cmap="viridis", marker=markerslist[k], label="Group "+markerslist[k])

  ax.set_xlabel(r"Projection sur $X'_1/\sigma'_1$")
  ax.set_ylabel(r"Projection sur $X'_2/\sigma'_2$")
  plt.legend()
  plt.show()

def get_DATA_2D_in_clusters(DATA, nb_clusters, verbose=None):
  """ Affiche la projection des individus dans l'espace des 3 variables d'inertie maximale avec clustering. """
  if nb_clusters > 26:
    print("Not enough markers to distinguish all the clusters.")
  labels, inertia = find_clusters(DATA, nb_clusters)
  print("Inertia :", inertia)
  return labels, inertia

def plot_DATA_2D_in_clusters(DATA, labels):
  K = np.max(labels)
  markerslist = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", r"$\mathcal{D}$", 
  r"$\mathcal{E}$", r"$\mathcal{F}$", r"$\mathcal{G}$", r"$\mathcal{H}$", r"$\mathcal{I}$", 
  r"$\mathcal{J}$", r"$\mathcal{K}$", r"$\mathcal{L}$", r"$\mathcal{M}$", r"$\mathcal{N}$",
  r"$\mathcal{O}$", r"$\mathcal{P}$", r"$\mathcal{Q}$", r"$\mathcal{R}$", r"$\mathcal{S}$",
  r"$\mathcal{T}$", r"$\mathcal{U}$", r"$\mathcal{V}$", r"$\mathcal{W}$", r"$\mathcal{X}$",
  r"$\mathcal{Y}$", r"$\mathcal{Z}$"]
  cols = ["blue", "orange", "green", "red", "purple", "grey", "brown", "pink", "purple", "cyan", "beige", "deeppink"]
    

  for k in range(K+1):  
    l_x = []
    l_y = []
    for i, label in enumerate(labels):
      if label==k:
        indiv = DATA[i]
        x1 = indiv[0]
        y1 = indiv[1]
        l_x.append(x1)
        l_y.append(y1)

    plt.scatter(l_x, l_y,cmap="viridis", marker="+",label="Groupe "+markerslist[k], color = cols[k]) #,edgecolor='black', linewidth='3')

  plt.gca().set_xlabel(r"Projection sur $X'_1/\sigma'_1$")
  plt.gca().set_ylabel(r"Projection sur $X'_2/\sigma'_2$")
  plt.legend()

def print_names_in_cluster(DATA, labels, names):
  out = {}
  for i in range(len(labels)):
    v = labels[i]
    if not v in out.keys():
      out[v] = []
    out[v] += [names[i]]

  """for j in out.keys():
    print("Cluster n0:"  +str(j))
    for v in out[j]:
      print("  " + str(v))"""
  
  return out
    
def show_images_from_names(names, folder, physicsdict, n, ext="fits", title=None, p=5):
  shuffle(names)
  iterm = min(len(names), n*n)
  curi = 0
  size_window = [7, 8]
  fig = plt.figure(figsize = (*size_window,))

  if title!=None:
    plt.suptitle(title)

  for i in range(iterm):
    v = names[i]
    key = key_name(v)
    z = physicsdict[str(key)]['PHOTOZ']
    M = physicsdict[str(key)]['ip_MAG_AUTO']
    F_R = physicsdict[str(key)]['FLUX_RADIUS']
    try:
      #print(v)
      img = get_image(folder + "/" + v + "." + ext)[0]
    except Exception as e:
      pass
    else:
      curi += 1
      plt.subplot(n,n, curi)
      plt.imshow(img, cmap="viridis")
      plt.annotate("z = "+str(z)+"\nM = "+str(M)+"\nF_R = "+str(F_R), xy=(10,55), xytext=(10,55), size=6, color='white')
      Center_of_Mass = scipy.ndimage.measurements.center_of_mass(img)
      formatted_CoM = Center_of_Mass[::-1]
      circle = plt.Circle(formatted_CoM, radius=p*F_R, fill=False, color="red")
      ax = fig.gca()
      ax.add_artist(circle)
    
  plt.tight_layout()
  plt.show()
