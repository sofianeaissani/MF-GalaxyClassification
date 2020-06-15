import numpy as np
import math

from astropy.io import fits
from scipy.signal import convolve as scipy_convolve
from scipy import signal
from astropy.convolution import Gaussian2DKernel
from random import *
from copy import copy
from PIL import Image



def contrast(file1):
  """
    A ne pas utiliser pour le moment, on doit trouver une meilleure fonction mathématique
  """
  return file1
  """file1 = np.float64(file1)
  file1 = ((file1 - 128) / 128) * (np.pi / 2)
  file1 = 1*np.tanh(file1) + 1
  file1 = (file1 * 128) / 1
  return file1"""

def contrastLinear(file1, value):
  file1 = value*file1
  return file1


def fantom(file1):
  """
    Enlève les pixels fantomes créés par le crop de ds9
  """
  file2 = []
  for i in range(len(file1)):
      file3 = []
      for c in range(len(file1[i])):
          if not np.isnan(file1[i][c]):
              file3 += [file1[i][c]]
      if file3 != []:
          file2 += [file3]
  file2 = np.float64(file2)
  return file2

def smooth_file(file1, size_gauss):
  """
    Lisse une image avec une matrice gaussiene
  """
  kernel = Gaussian2DKernel(size_gauss)
  file1 = scipy_convolve(file1, kernel, mode='same', method='direct')
  return file1

def get_dat_file(name):
  """ A 'dat' file is here simply a matrix encoded in a file """ 
  file1 = np.loadtxt(name)
  file1 = np.float64(file1)
  return file1

def get_fit_file(name):
  file1 = fits.getdata(name)
  file1 = np.float64(file1)
  return file1

def charger_fichier_A(path):
  """ Charge les agréables fichiers de Carlo qui sont sous la forme de plusieurs colonnes """

  matrix = []
  file_r = open(path, "r")

  for line in file_r:
    temp = line.split()
    if temp[0] != "#":
        x = int(temp[2])-1
        y = int(temp[3])-1  
        v = float(temp[4])
        if y >= len(matrix):
          matrix.append([])
        if x >= len(matrix[y]):
          matrix[y].append([])
        matrix[y][x] = v
  matrix = np.float64(matrix)
  name = path.split("/")[-1].split(".")[0]
  return matrix, name

def get_image(path, override=""):
  """
    Obtenir une image en format np-array-64, son nom et son extension
  """
  name = path.split("/")
  name = name[-1]
  ext = ""

  if "." in name:
    name = name.split(".")
    ext = name[-1]
    name = name[-2]


  # Récupérer le fichier
  if override:
    if override == "A":
      file1,name = charger_fichier_A(path)
      ext = ""
    else:
      pass
  else:
    if ext == "dat":
      file1 = get_dat_file(path)
    elif ext == "fits":
      file1 = get_fit_file(path)
      file1 = fantom(file1)
  file1 = cool_range(file1)
  return file1,name,ext

def degrade(file1, val):
    """ Dégrade la qualité d'une image en diminuant son nombre de pixels, val est le facteur de division """
    assert type(val) == int
    img1 = np.float64(file1)
    img2 = []
    for i in range(math.floor(img1.shape[0]/val)):
        img2 += [[]]
        for j in range(math.floor(img1.shape[1]/val)):
            moyenne = (img1[i*val][j*val] + img1[i*val + 1][j*val] + img1[i*val][j*val + 1] + img1[i*val + 1][j*val + 1])/(val**2)
            img2[i] += [moyenne]
    img2 = np.float64(img2)
    return img2
  
def degradePIL(file1, val):
    """ Dégrade la qualité d'une image en diminuant son nombre de pixels, val est le facteur de division """
    assert type(val) == int
    img1 = np.float64(file1)
    img2 = Image.fromarray(img1)
    img2 = img2.resize((math.floor(img1.shape[1] / val), math.floor(img1.shape[0] / val)), resample=0)
    img2 = np.float64(img2)
    return img2


def pepper_and_salt(file2, pourcentage):
    file1 = file2.copy()
    for i in range(len(file1)):
        for j in range(len(file1[i])):
            lepourcentagealeatoire = random()
            if lepourcentagealeatoire <= pourcentage:
                lenombrealeatoire = randint(0, 1)
                if lenombrealeatoire == 0:
                    file1[i][j] = 0
                elif lenombrealeatoire == 1:
                    file1[i][j] = 255
    return file1


def adaptive_poisson_noise(img, coef, truncate=False): 
  noise_mask = np.random.poisson(img*coef)/coef # example : coef = 1
  #noisy_img = img + noise_mask 
  # return the mask instead of adding the mask to the image 
  # (in order to allow losses of luminosity on some pixels)
  if truncate:
    return np.clip(noise_mask, 0, 255) 
  else:
    return noise_mask

def uniform_poisson_noise(img, parameter, truncate=False):
  noise_mask = np.random.poisson(parameter, img.shape) # example : parameter = 25
  noisy_img = img + noise_mask - parameter*np.ones(np.shape(img)) # the noisy image is more luminous than the initial one
  if truncate:
    return np.clip(noisy_img, 0, 255)
  else:
    return noisy_img

def gaussian_noise(img, std):
  noise_mask = np.random.normal(0, std, img.shape)  # std : standard deviation
  noisy_img = img + noise_mask
  return noisy_img

def rotation_X(img,theta):
  img2=[[[] for i in range(len(img[0]))] for j in range(len(img))]
  for y,line in enumerate(img):
    for x, val in enumerate(line):
      #print(y,x)
      xx = x
      yy = (y-len(img)//2)*math.cos(theta)+len(img)//2
      #print(yy)
      target = img2[int(yy)][int(xx)]
      img2[int(yy)][int(xx)].append(val)
  #print(img2[1][1])
  for y,line in enumerate(img2):
    for x, val in enumerate(line):
      #print(y,x)
      if img2[y][x]==[]:
        img2[y][x] = 0
      else:
        img2[y][x] = np.mean(img2[y][x])

  img2 = np.float64(img2)

  return img2

def cool_range(matrix):
  m = matrix.min()
  matrix = matrix - m
  matrix = 255*matrix/matrix.max()
  return matrix
  
def quadrimean(img,x,y):
  summ = 0
  for x,y in [ [int(x),int(y)], [int(x)+1,int(y)],[int(x)+1,int(y)+1],[int(x),int(y)+1] ]:
    if y < len(img) and x < len(img[y]):
      summ += img[y][x]
  return summ/4

def second_inflexion_point(file1):

  file1 = np.clip(np.rint(file1), 0, 255)
    
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

  threshold = np.argmin(secondSmooth)
  while secondSmooth[threshold+1] < 0:
    threshold += 1

  print(threshold)

  file2 = file1.copy()
  file2[file2 < threshold] = threshold

  return file2
