# MF-GalaxyClassification

## File tree

| File | Content | Type |
| :-----: | :-----: | :-----: |
|   |   |   | 
| libs/imProcess.py | Image processing (noise, contrast, smoothing, etc) | Library |
| libs/MF.py | Minkowski Functionals computing | Library |
| libs/matProcess.py | Matrix building, PCA processing, graphics | Library |
|   |   |   | 
| mainMF.py | Computes and displays the Minkowski Functions of the input image (supported : .fits, .dat) | Program |
| mfSmoothTransitionUnique.py | Displays the chosen Minkowski Function of an image for different smooth levels | Program |
| mfSmoothTransitionAll.py | Displays all the Minkowski Functions of an image for different smooth levels | Program |
| mfInteractive.py | Displays the binarized images corresponding to an input image in an interactive way | Program |
| mfEvery.py | Displays the chosen Minkowski Function of all the images in a directory | Program |
|   |   |   |
| mainCAS.py | Computes C, A, S and other indicators (<img src="https://render.githubusercontent.com/render/math?math=M_{20}">, Gini...) on the input image | Program |
|   |   |   |
| noiseDisplay.py | Displays the effect of different types of noise on the input image | Program |
| noiseByType.py | Compares the effect of noise by type on the input image | Program |
| noiseByStrength.py | Compares the effect of the chosen noise by strength on the input image | Program |
| noiseFindThreshold.py | Finds the intensity level of the 2nd inflexion point of the smoothed intensity histogram of the input image | Program |
|   |   |   |
| pcaClustering.py | Executes the PCA and the <img src="https://render.githubusercontent.com/render/math?math=k">-means clustering on the images of the input directory | Program |
| pcaClustering2.py | Same, but it also prints the names of all the images of each cluster | Program |
|   |   |   | 
| plotClustersFromTxt.py | Plots a <img src="https://render.githubusercontent.com/render/math?math=n\times n"> sample of the images of the input .txt file | Program |
| cropImages.py | Displays and crops all the images of the input directory according to user inputs | Program |
| thresholdDisplay.py | Displays some binarized images corresponding to an input image | Program |

| **Warning : modifying the "Library" files can lead to a lot of errors in other programs.** |
| --- |

### mainMF.py
  `python3 main.py [FITS or DAT file] [optional args]`

  - The path to the image can be absolute or relative.
  - The optional arguments and their use can be seen typing `python3 main.py --h`


### mainCAS.py
  `python3 morpho.py [FITS ou DAT file] [optional args]`

### pcaClustering.py

  - One must give as an argument the directory containing all the images.
  - The data matrix can be saved in a file using `-s npy/datamatrix.npy` and can be fetched using `-l npy/datamatrix.npy`.