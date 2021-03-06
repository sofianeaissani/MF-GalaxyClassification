# MF-GalaxyClassification

## Requirements (owner's setup)

- Python 3.7
- minkfncts2d 1.0 (https://github.com/moutazhaq/minkfncts2d)
- astropy 4.0
- numpy 1.18.2
- matplotlib 3.2.0rc1
- imageIO 2.6.1
- photutils 0.7.2
- Pillow 5.2.0
- scikit-image 0.16.2
- scikit-learn 0.23.1
- scipy 1.3.0
- statmorph 0.3.5 (https://github.com/vrodgom/statmorph)

## File tree

| File | Content | Type |
| :-----: | :-----: | :-----: |
|   |   |   | 
| libs/imProcess.py | Image processing (noise, contrast, smoothing, etc) | Library |
| libs/MF.py | Minkowski Functionals computing | Library |
| libs/matProcess.py | Matrix building, PCA processing, graphics | Library |
|   |   |   | 
| mainMF.py | Computes and displays the MF of the input image (supported : .fits, .dat) | Program |
| mfSmoothTransitionUnique.py | Displays the chosen MF of an image for different smooth levels | Program |
| mfSmoothTransitionAll.py | Displays all the MF of an image for different smooth levels | Program |
| mfInteractive.py | Displays the binarized images corresponding to an input image in an interactive way | Program |
| mfEvery.py | Displays the chosen MF of all the images in a directory | Program |
|   |   |   |
| mainCAS.py | Computes C, A, S and other indicators (<img src="https://render.githubusercontent.com/render/math?math=M_{20}">, Gini...) on the input image | Program |
|   |   |   |
| noiseDisplay.py | Displays the effect of different types of noise on the input image | Program |
| noiseByType.py | Compares the effect of noise by type on the input image | Program |
| noiseByStrength.py | Compares the effect of the chosen noise by strength on the input image | Program |
| noiseFindThreshold.py | Finds the intensity level of the 2nd inflexion point of the smoothed intensity histogram of the input image | Program |
|   |   |   |
| pcaClustering.py | Runs PCA and <img src="https://render.githubusercontent.com/render/math?math=k">-means clustering on the images of the input directory | Program |
| pcaClustering2.py | Same, but it also prints the names of all the images of each cluster | Program |
|   |   |   | 
| plotClustersFromTxt.py | Plots a <img src="https://render.githubusercontent.com/render/math?math=n\times n"> sample of the images of the input .txt file | Program |
| cropImages.py | Crops all the images of the input directory according to user inputs | Program |
| thresholdDisplay.py | Displays some binarized images corresponding to an input image | Program |

| **Warning : modifying the "Library" files can lead to a lot of errors in other programs.** |
| --- |

### mainMF.py
  `python3 mainMF.py [FITS or DAT file] [optional args]`

  - The path to the image can be absolute or relative.
  - The optional arguments and their use can be seen typing `python3 mainMF.py --h`


### mainCAS.py
  `python3 mainCAS.py [FITS ou DAT file] [optional args]`

### pcaClustering.py

  - One must give as an argument the directory containing all the images.
  - The data matrix can be saved in a file using `-s npy/datamatrix.npy` and can be fetched using `-l npy/datamatrix.npy`.