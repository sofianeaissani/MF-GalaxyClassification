import libs.matrices3 as mt
from random import shuffle

def main():
  a = open("clusterfiles/dataset1/0.txt")
  lib = []
  folder = "data/dataset1_z075-100_M214"
  for line in a:
    lib.append(line.replace("\n", "").replace(' ', '')+".fits")
  shuffle(lib)
  mt.show_images_from_names(lib, folder, 3)

if __name__ == "__main__":
    main()