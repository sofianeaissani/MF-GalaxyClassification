import os, shutil, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from astropy.nddata import Cutout2D

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS

a = os.listdir()

plt.ion()
if not "michel" in a:
    os.mkdir("michel")

def fantom(file1):
  """
    Enleve les pixels fantomes créés par le crop de ds9
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


def save_cutout2(orig,pos, size, out, lamatrice):
    hdu = fits.open(orig)[0]
    wcs = WCS(hdu.header)

    # Make the cutout, including the WCS
    cutout = Cutout2D(lamatrice, position=pos, size=size, wcs=wcs)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    # Write the cutout to a new FITS file
    hdu.writeto(out, overwrite=True)

def save_cutout(orig,pos, size, out, lamatrice):
  # hdu = fits.open(orig)[0]
  # wcs = WCS(hdu.header)

  # Make the cutout, including the WCS
  # cutout = Cutout2D(lamatrice, position=pos, size=size, wcs=wcs)

  # Put the cutout image in the FITS HDU
  # hdu.data = cutout.data

  # Update the FITS header with the cutout WCS
  # hdu.header.update(cutout.wcs.to_header())

  # Write the cutout to a new FITS file
  # hdu.writeto(out, overwrite=True)
  hdu = fits.PrimaryHDU(lamatrice)
  hdu.writeto(out, overwrite=True)



def crop_matrix(m, x1, y1,x2,y2):
  new_m = []
  for i in range(len(m)):
    if i > x1 and i<x2:
      cur_line = []
      for j in range(len(m[i])):
        if j > y1 and j<y2:
          cur_line += [m[i][j]]
      new_m += [cur_line]
  return new_m

def plot_image(img):
  plt.close()
  plt.imshow(img, cmap='viridis')
  plt.xlabel("axe des Y")
  plt.ylabel("axe des X")
  plt.show()
for i in a:
  if i[-3:] == "its":
    file1 = fits.getdata(i)
    file1 = np.float64(file1)
    file1 = fantom(file1)
    plot_image(file1)
    b = ""
    while b not in ["o", "n", "c"]:
      b = input("")
    if b == "o":
      print(i)
      shutil.copyfile(i, "./michel/"+i)
    elif b == "n":
      pass
    elif b == "c":
      done = False
      while done == False:
        try:
          temp = input("x1,y1:\n").split()
          x1 = int(temp[0])
          y1 = int(temp[1])
          temp = input("x2,y2:\n").split()
          x2 = int(temp[0])
          y2 = int(temp[1])
          new_m = crop_matrix(file1, x1,y1,x2,y2)
          new_m = np.float64(new_m)
          plot_image(new_m)
          c = ""
          while c not in ["o", "n"]:
            c = input("")
          if c == "o":
            done = True
            save_cutout(i, (x1,y1), (x2-x1,y2-y1), "./michel/"+i, new_m)
          else:
            plot_image(file1)

        except Exception as e:
          print(e)
      
    plt.close()


def crop_matrix(m, x1, y1,x2,y2):
  new_m = []
  for i in range(len(m)):
    if i > x1 and i<x2:
      cur_line = []
      for j in range(len(m[i])):
        if j > y1 and j<y2:
          cur_line += [m[i][j]]
      new_m += [cur_line]
  return new_m
