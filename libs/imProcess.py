import numpy as np
import matplotlib.pyplot as plt
import math

from astropy.io import fits
from scipy.signal import convolve as scipy_convolve
from scipy import signal
from astropy.convolution import Gaussian2DKernel
from random import *
from copy import copy
from PIL import Image

from libs.phyProcess import key_name


def remove_fantom_pixels(img): #### TO BE MODIFIED
  """
    Removes all NaN pixels from the image
  """
  file2 = []
  for i in range(len(img)):
      file3 = []
      for c in range(len(img[i])):
          if not np.isnan(img[i][c]):
              file3 += [img[i][c]]
      if file3 != []:
          file2 += [file3]
  file2 = np.float64(file2)
  return file2

def smooth_img(img, size_gauss):
  """
    Smoothes the image by convolution with a gaussian kernel
  """
  kernel = Gaussian2DKernel(size_gauss)
  img = scipy_convolve(img, kernel, mode='same', method='direct')
  return img

def get_dat_file(myFile):
  """ A 'dat' file is here simply a matrix encoded in a file """ 
  file1 = np.loadtxt(myFile)
  img = np.float64(file1)
  return img

def get_fits_file(myFile):
  file1 = fits.getdata(myFile)
  img = np.float64(file1)
  return img

def get_dat_file2(myFile): #### TO BE MODIFIED
  """ Charge les agréables fichiers de Carlo qui sont sous la forme de plusieurs colonnes """

  matrix = []
  file_r = open(myFile, "r")

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
  name = myFile.split("/")[-1].split(".")[0]
  return matrix, name

def get_image(myFile, override=""):
  """
    Imports an image, its name and its extension from the given path
  """
  name = myFile.split("/")
  name = name[-1]
  ext = ""

  if "." in name:
    name = name.split(".")
    ext = name[-1]
    name = name[-2]


  # Récupérer le fichier
  if override:
    if override == "A":
      img, name = get_dat_file2(myFile)
      ext = ""
    else:
      pass
  else:
    if ext == "dat":
      img = get_dat_file(myFile)
    elif ext == "fits":
      img = get_fits_file(myFile)
      #img = remove_fantom_pixels(img)

  return img,name,ext

def get_image_and_properties(myFile, DICT, override=""):
  """
    Imports an image, its name, its extension and its properties from the given path
  """
  name = myFile.split("/")
  name = name[-1]
  ext = ""

  if "." in name:
    name = name.split(".")
    ext = name[-1]
    name = name[-2]


  # Récupérer le fichier
  if override:
    if override == "A":
      img, name = get_dat_file2(myFile)
      ext = ""
    else:
      pass
  else:
    if ext == "dat":
      img = get_dat_file(myFile)
    elif ext == "fits":
      img = get_fits_file(myFile)
      #img = remove_fantom_pixels(img)

  key = key_name(myFile)
  properties = DICT[key]

  return img,name,ext,properties

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

"""def second_inflexion_point(file1, cutoutsize):


  arrays = np.array_split(file1, 10)

  for array in arrays:

    numin = np.min(array)
    numax = np.max(array)
      
    NbOccurs = []
    threshold = [numin + i*(numax-numin)/300 for i in range(300)]
    for i in threshold:
      Occurs_i = np.count_nonzero(array == i)
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

    array[array < threshold] = threshold
    array = array - threshold

  file2 = np.vstack([array for array in arrays])
  file2 = np.reshape(file2, np.shape(file1))

  return file2"""

def second_inflexion_point(image, cutout_size):

  img = np.copy(image)

  n,m = np.shape(img)[0], np.shape(img)[1]

  i = 0
  j = 0
  
  kernel = signal.gaussian(28, 7)

  while i + cutout_size < n:
    while j + cutout_size < m:
      

      ### Initialize temporary table
      temp = img[i:i+cutout_size, j:j+cutout_size]
      numin = np.min(temp)
      numax = np.max(temp)

      ### Make the intensity histogram
      NbOccurs = []
      threshold = [numin + k*(numax-numin)/300 for k in range(300)]
      
      for k in threshold:
        Occurs = np.count_nonzero(np.logical_and(k <= temp, temp < k + (numax-numin)/300))
        NbOccurs.append(Occurs)
      
      NbOccursSmooth = np.convolve(NbOccurs, kernel, mode='same')
      accroiss = np.diff(NbOccursSmooth, append=[0])
      accroissSmooth = np.convolve(accroiss, kernel, mode='same')
      second = np.diff(accroissSmooth, append=[0])
      secondSmooth = np.convolve(second, kernel, mode='same')

      k = np.argmin(secondSmooth)
      while secondSmooth[k] < 0:
        k += 1
      critical_threshold = threshold[k]

      temp[temp < critical_threshold] = critical_threshold
      temp = temp - critical_threshold

      img[i:i+cutout_size, j:j+cutout_size] = temp
    
      j += cutout_size
    i += cutout_size
    if i + cutout_size < n: j = 0

  ### right side

  temp = img[:i, j:]
  numin = np.min(temp)
  numax = np.max(temp)


  NbOccurs = []
  threshold = [numin + k*(numax-numin)/300 for k in range(300)]
  for k in threshold:
    Occurs = np.count_nonzero(np.logical_and(k <= temp, temp < k + (numax-numin)/300))
    NbOccurs.append(Occurs)
 
  NbOccursSmooth = np.convolve(NbOccurs, kernel, mode='same')
  accroiss = np.diff(NbOccursSmooth, append=[0])
  accroissSmooth = np.convolve(accroiss, kernel, mode='same')
  second = np.diff(accroissSmooth, append=[0])
  secondSmooth = np.convolve(second, kernel, mode='same')

  k = np.argmin(secondSmooth)
  while secondSmooth[k] < 0:
    k += 1
  critical_threshold = threshold[k]

  temp[temp < critical_threshold] = critical_threshold
  temp = temp - critical_threshold

  img[:i, j:] = temp
      
  ### bottom side and right-bottom side

  temp = img[i:, :]
  numin = np.min(temp)
  numax = np.max(temp)


  NbOccurs = []
  threshold = [numin + k*(numax-numin)/300 for k in range(300)]
  for k in threshold:
    Occurs = np.count_nonzero(np.logical_and(k <= temp, temp < k + (numax-numin)/300))
    NbOccurs.append(Occurs)

  NbOccursSmooth = np.convolve(NbOccurs, kernel, mode='same')
  accroiss = np.diff(NbOccursSmooth, append=[0])
  accroissSmooth = np.convolve(accroiss, kernel, mode='same')
  second = np.diff(accroissSmooth, append=[0])
  secondSmooth = np.convolve(second, kernel, mode='same')

  k = np.argmin(secondSmooth)
  while secondSmooth[k] < 0:
    k += 1
  critical_threshold = threshold[k]

  temp[temp < critical_threshold] = critical_threshold
  temp = temp - critical_threshold

  img[i:, :] = temp

  return img

def normalize(image):
        m, M = np.min(image), np.max(image)
        return (image-m) / (M-m)
