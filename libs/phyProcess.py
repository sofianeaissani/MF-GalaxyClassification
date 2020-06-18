from astropy.io import fits
import numpy as np
import sys
import pandas as pd


def extract_galaxies_data(csv_file):
  file1 = open(csv_file)
  keys = []
  final = {}
  for i,line in enumerate(file1):
    if i == 0:
      keys = line.split(",")
      keys[-1] = keys[-1].replace("\n", "")
    else:
      temp = line.split(",")
      temp[-1] = temp[-1].replace("\n", "")
      dico = {}
      key_name = temp[0].replace(".", "p") +"_"+temp[1].replace(".", "p")
      for j,v in enumerate(keys):
        dico[v] = float(temp[j])
      final[key_name] = dico
  return final

def key_name(galaxy_file):
  temp = galaxy_file.split("_")
  return temp[1] + "_" + temp[2]

def export_groups_to_TOPCAT(clustersdict, physicsdict, output_path):
    clustersArrays = []
    example_key = list(physicsdict.keys())[0]
    for group_index in range(len(list(clustersdict.keys()))):
        group = clustersdict[group_index]
        for i, name in enumerate(group):
            if i == 0:
                key = key_name(name)
                clusterArray = np.array(list(physicsdict[key].values()))
            else:
                key = key_name(name)
                clusterArray = np.vstack((clusterArray, np.array(list(physicsdict[key].values()))))
        clustersArrays.append(clusterArray)
    for i, group in enumerate(clustersArrays):
        columns = []
        for j in range(np.shape(clustersArrays[i])[1]):
            columns.append(fits.Column(name=list(physicsdict[example_key].keys())[j], array=clustersArrays[i][:,j], format='D'))
        t = fits.BinTableHDU.from_columns(columns)
        t.writeto(output_path+'_Group'+chr(65+i)+".fits")


    
